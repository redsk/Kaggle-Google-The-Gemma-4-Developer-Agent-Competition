"""Offline checks against organizer assets: no model, container or network calls."""

import asyncio
import ast
from dataclasses import asdict
import json
import os
from pathlib import Path
import shlex
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.update({"LITELLM_LOCAL_MODEL_COST_MAP": "True", "LITELLM_TELEMETRY": "False",
                   "OTEL_SDK_DISABLED": "true", "HF_HUB_OFFLINE": "1"})

import yaml
from adk_submission import compile_submission, VllmConfig, VllmServer
from google.adk.apps.llm_event_summarizer import LlmEventSummarizer
from google.adk.events import Event
from google.adk.models.lite_llm import LiteLLMClient
from google.adk.models.llm_request import LlmRequest
from google.genai import types
from litellm import ModelResponse
from swegemma.budget import HarnessLimits
from swegemma.config import build_submission_limits
from swegemma.context import SwegemmaContext

from run_supervised import Monitor, build_eval_config, describe_config, validate_server_contexts
from vllm_client import remaining_context_parameters, VllmRemainingContextClient


class ChallengeConfigurationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.args = SimpleNamespace(minutes=None, baseline=False, pause_on_alert=False,
                                   models_yaml=ROOT / "dev/models.vllm.yaml")
        cls.config = build_eval_config(cls.args, ROOT / "results/offline-config-check")
        cls.local_agent = yaml.safe_load((ROOT / "submission/agent.yaml").read_text())
        cls.official_sampling = yaml.safe_load(
            (ROOT / "downloads/competition/sample_submission/configs/sampling.yaml").read_text())

    def test_sampling_uses_permitted_maxima_and_official_sampling_defaults(self):
        _, constraints = build_submission_limits()
        expected = dict(self.official_sampling)
        expected["max_output_tokens"] = constraints.max_output_tokens.max
        expected["thinking_config"] = dict(expected["thinking_config"], thinking_budget=constraints.thinking_budget.max)
        self.assertEqual(self.local_agent["generate_content_config"], expected)

    def test_tool_limits_and_testing_match_harness(self):
        self.assertEqual(self.config.harness, HarnessLimits())
        self.assertTrue(self.config.enable_sandbox_testing)
        self.assertNotIn("Keep reasoning brief", self.local_agent["instruction"])
        self.assertNotIn("Pytest and broad test discovery are disabled", self.local_agent["instruction"])
        context = SwegemmaContext(budget=self.config.budget, harness=self.config.harness)
        self.assertEqual(set(self.local_agent["tools"]), set(context.create_tools()))

    def test_no_added_task_budget(self):
        # User explicitly requested no custom task time, turn or tool-call caps.
        self.assertTrue(all(v is None for v in asdict(self.config.budget).values()))
        self.assertFalse(self.args.pause_on_alert)
        json.dumps(describe_config(self.config, self.args))

    def test_context_management_matches_organizer_notebook(self):
        notebook = json.loads((ROOT / "downloads/starter/getting-started-gemma-4-developer-agent.ipynb").read_text())
        expected = {}
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            for node in ast.walk(ast.parse("".join(cell["source"]))):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {"ContextCacheConfig", "EventsCompactionConfig"}:
                    expected[node.func.id] = {k.arg: ast.literal_eval(k.value) for k in node.keywords}
        self.assertEqual(self.config.context_cache_config.model_dump(exclude_none=True), expected["ContextCacheConfig"])
        self.assertEqual(self.config.events_compaction_config.model_dump(exclude_none=True), expected["EventsCompactionConfig"])

    def test_remaining_context_mode_rejects_wrong_server_context(self):
        mapping = yaml.safe_load(self.args.models_yaml.read_text())
        entry = next(iter(mapping["models"].values()))
        model_id = entry["path"].removeprefix("openai/")
        limit = self.config.generation_constraints.max_output_tokens.max
        response = SimpleNamespace(raise_for_status=lambda: None,
                                   json=lambda: {"data": [{"id": model_id, "max_model_len": limit}]})
        with patch("requests.get", return_value=response):
            self.assertEqual(len(validate_server_contexts(mapping, limit)), 1)
            with self.assertRaises(ValueError):
                validate_server_contexts(mapping, limit - 1)

    def test_real_compiler_and_client_forward_generation_settings(self):
        context = SwegemmaContext(budget=self.config.budget, harness=self.config.harness)
        agent = compile_submission(
            self.config.submission_dir, tool_registry=context.create_tools(),
            model_registry=self.config.models, limits=self.config.limits,
            generation_constraints=self.config.generation_constraints,
        )
        captured = []

        async def capture(self, model, messages, tools, **kwargs):
            captured.append(kwargs)
            return ModelResponse(model=model, choices=[{
                "index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": "Synthetic response."},
            }], usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2})

        self.assertIsInstance(agent.model.llm_client, VllmRemainingContextClient)
        agent.model.llm_client.log_path = None

        async def exercise():
            content = types.Content(role="user", parts=[types.Part(text="Synthetic request.")])
            request = LlmRequest(contents=[content], config=agent.generate_content_config)
            async for _ in agent.model.generate_content_async(request):
                pass
            await LlmEventSummarizer(agent.model).maybe_summarize_events(
                events=[Event(author="user", content=content)])

        with patch.object(LiteLLMClient, "acompletion", capture):
            asyncio.run(exercise())
        self.assertEqual(len(captured), 2)
        self.assertEqual(agent.generate_content_config.max_output_tokens,
                         self.config.generation_constraints.max_output_tokens.max)
        self.assertNotIn("max_completion_tokens", captured[0])
        self.assertNotIn("max_tokens", captured[0])
        self.assertEqual(captured[0]["extra_body"]["thinking_token_budget"],
                         self.config.generation_constraints.thinking_budget.max)
        self.assertTrue(captured[0]["extra_body"]["chat_template_kwargs"]["enable_thinking"])
        # The official summarizer doesn't supply an output ceiling. Do not add one.
        self.assertNotIn("max_completion_tokens", captured[1])
        self.assertNotIn("max_tokens", captured[1])

    def test_explicit_smaller_requests_are_preserved(self):
        limit = self.config.generation_constraints.max_output_tokens.max
        request = {"max_completion_tokens": 128, "temperature": 0.2}
        self.assertEqual(remaining_context_parameters(request, limit), request)
        self.assertEqual(request["max_completion_tokens"], 128)
        self.assertEqual(remaining_context_parameters({}, limit), {})
        self.assertEqual(remaining_context_parameters({"max_completion_tokens": limit, "max_tokens": 128}, limit), {})
        self.assertEqual(remaining_context_parameters({"max_tokens": limit}, limit), {})
        with self.assertRaises(ValueError):
            remaining_context_parameters({"max_completion_tokens": limit + 1}, limit)

    def test_launch_settings_match_official_server_builder(self):
        text = (ROOT / "scripts/start_vllm.sh").read_text().replace("\\\n", " ")
        local = shlex.split(text.split("exec ", 1)[1])[2:]
        official = VllmServer(VllmConfig(model="unused", tool_call_parser="gemma4",
                                         reasoning_parser="gemma4", enable_lora=False)).build_cmd()
        for flag in ("--max-model-len", "--gpu-memory-utilization", "--tool-call-parser", "--reasoning-parser"):
            ours, theirs = local[local.index(flag)+1], official[official.index(flag)+1]
            self.assertEqual(float(ours) if flag == "--gpu-memory-utilization" else ours,
                             float(theirs) if flag == "--gpu-memory-utilization" else theirs)
        self.assertEqual(json.loads(local[local.index("--reasoning-config")+1]),
                         json.loads(official[official.index("--reasoning-config")+1]))
        self.assertIn("--no-scheduler-reserve-full-isl", local)
        for flag in ("--override-generation-config", "--generation-config", "--max-num-seqs", "--max-num-batched-tokens"):
            self.assertNotIn(flag, local)
        generation = json.loads((ROOT / "models/vllm/gemma-4-12B-it-qat-4bit/generation_config.json").read_text())
        self.assertNotIn("max_new_tokens", generation)

    def test_monitor_alerts_do_not_pause_by_default(self):
        with tempfile.TemporaryDirectory() as directory:
            monitor = Monitor(Path(directory), SimpleNamespace(container_id=""))
            for _ in range(4):
                monitor.on_entry(SimpleNamespace(
                    event_type="continuation_nudge", to_dict=lambda: {"event_type": "continuation_nudge"}))
            asyncio.run(asyncio.wait_for(monitor.before_model(SimpleNamespace(contents=[])), timeout=1))
            self.assertEqual(monitor.calls, 1)
            self.assertFalse((Path(directory) / "pause.json").exists())

    def test_opt_in_monitor_pause_still_works(self):
        with tempfile.TemporaryDirectory() as directory:
            monitor = Monitor(Path(directory), SimpleNamespace(), pause_on_alert=True)
            monitor.alert("Explicitly enabled review pause")
            self.assertTrue((Path(directory) / "pause.json").exists())


if __name__ == "__main__":
    unittest.main()
