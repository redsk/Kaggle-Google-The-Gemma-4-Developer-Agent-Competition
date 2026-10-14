import asyncio
import os
from pathlib import Path
import sys
import tempfile
import unittest
import json
from unittest.mock import patch
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.update({"LITELLM_LOCAL_MODEL_COST_MAP": "True", "LITELLM_TELEMETRY": "False", "OTEL_SDK_DISABLED": "true"})

from observability_fixture import exercise
from observability import Telemetry, instrument


class ObservabilityTest(unittest.TestCase):
    def test_remote_registry_preserves_explicit_model_and_effort(self):
        from adk_submission import compile_submission
        from google.adk.models.lite_llm import LiteLLMClient
        from google.adk.models.llm_request import LlmRequest
        from google.genai import types
        from litellm import ModelResponse
        from swegemma.context import SwegemmaContext
        from run_supervised import build_eval_config
        args = SimpleNamespace(minutes=None, baseline=False, pause_on_alert=False,
                               models_yaml=ROOT / "dev/models.vllm.yaml", backend="cliproxy",
                               remote_model="gpt-6.1-sol", reasoning_effort="low")
        captured = []
        async def capture(self, model, messages, tools, **kwargs):
            captured.append((model, kwargs))
            return ModelResponse(model=model, choices=[{"index": 0, "finish_reason": "stop",
                                  "message": {"role": "assistant", "content": "fixture"}}])
        with tempfile.TemporaryDirectory() as directory, \
             patch("cliproxy.client_settings", return_value=("http://127.0.0.1:8317/v1", "fixture-secret")), \
             patch.object(LiteLLMClient, "acompletion", capture):
            config = build_eval_config(args, Path(directory))
            context = SwegemmaContext(budget=config.budget, harness=config.harness)
            agent = compile_submission(config.submission_dir, tool_registry=context.create_tools(),
                                       model_registry=config.models, limits=config.limits,
                                       generation_constraints=config.generation_constraints)
            async def call():
                request = LlmRequest(config=agent.generate_content_config, contents=[types.Content(role="user", parts=[types.Part(text="fixture")])])
                async for _ in agent.model.generate_content_async(request):
                    pass
            asyncio.run(call())
        self.assertEqual(captured[0][0], "openai/gpt-6.1-sol")
        self.assertEqual(captured[0][1]["reasoning_effort"], "low")
        self.assertEqual(captured[0][1]["max_completion_tokens"], 32768)
        self.assertEqual(captured[0][1]["num_retries"], 5)

    def test_cancelled_generation_closes_span_and_restores_wrappers(self):
        # Official harness imports apply their own compatibility wrappers.
        # Establish the baseline after that initialization, as a task run does.
        import importlib
        importlib.import_module("swegemma.evaluate")
        from google.adk.models.lite_llm import LiteLlm, LiteLLMClient
        from google.adk.models.llm_request import LlmRequest
        from google.genai import types
        original = LiteLlm.generate_content_async
        with tempfile.TemporaryDirectory() as directory:
            obs = Telemetry(directory, endpoint=None)
            async def check():
                started = asyncio.Event()
                async def pending(self, *args, **kwargs):
                    started.set()
                    await asyncio.Event().wait()
                async def collect():
                    model = LiteLlm(model="openai/gemma4-12b-local", api_key="fixture", api_base="http://fixture.invalid/v1")
                    async for _ in model.generate_content_async(LlmRequest(contents=[types.Content(role="user", parts=[types.Part(text="cancel")])])):
                        pass
                with patch.object(LiteLLMClient, "acompletion", pending), obs.run(), instrument():
                    task = asyncio.create_task(collect())
                    await asyncio.wait_for(started.wait(), timeout=5)
                    task.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await task
            asyncio.run(check())
            self.assertIs(LiteLlm.generate_content_async, original)
            spans = [json.loads(line) for line in (Path(directory) / "spans.jsonl").read_text().splitlines()]
            model_span = next(s for s in spans if s["name"] == "llm.agent")
            self.assertEqual(model_span["status"]["description"], "CancelledError")

    def test_real_adk_pipeline_and_all_capture_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            result = asyncio.run(exercise(directory))
            self.assertEqual(result["status"], "passed")
            self.assertEqual(os.environ["OTEL_SDK_DISABLED"], "true")

    def test_nested_secret_redaction_and_full_content(self):
        with tempfile.TemporaryDirectory() as directory:
            obs = Telemetry(directory, endpoint=None, secrets=["known-secret-value"])
            with obs.run():
                data = {"Authorization": "Bearer example", "nested": [{"api_key": "anything"}],
                        "content": "x" * 50000 + "known-secret-value", "prompt_tokens": 27}
                result = obs.clean(data)
                self.assertEqual(result["Authorization"], "[REDACTED]")
                self.assertEqual(result["nested"][0]["api_key"], "[REDACTED]")
                self.assertEqual(result["content"], "x" * 50000 + "[REDACTED]")
                self.assertEqual(result["prompt_tokens"], 27)


if __name__ == "__main__":
    unittest.main()
