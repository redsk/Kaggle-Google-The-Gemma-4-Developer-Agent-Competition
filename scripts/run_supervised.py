"""Single-task local vLLM runner with live monitoring and a resumable pause gate.

Create pause.json in the result directory to pause before the next model call.
Remove it only after the user approves resuming. Optional user-approved guidance
in steering.txt is sent on resume. Raw task content stays in local log files.
"""

import argparse
import asyncio
from collections import Counter, defaultdict, deque
import contextlib
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from urllib.parse import urlparse

import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "downloads/competition"


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


class Monitor:
    def __init__(self, directory, ctx):
        self.directory, self.ctx = directory, ctx
        self.pending = defaultdict(deque)
        self.failures, self.repeats = Counter(), Counter()
        self.calls = self.tools = self.nudges = self.last_progress = self.changed_files = 0
        self.diff_hash = None
        self.paused = False

    def emit(self, event, **fields):
        record = {"time": datetime.now(timezone.utc).isoformat(), "event": event, **fields}
        line = json.dumps(record, default=str)
        with (self.directory / "progress.jsonl").open("a") as output:
            output.write(line + "\n")
        print(line, file=sys.__stdout__, flush=True)

    def pause(self, reason):
        path = self.directory / "pause.json"
        if not path.exists():
            save_json(path, {"reason": reason, "model_calls": self.calls, "tools": self.tools})
            self.emit("pause_requested", reason=reason)

    def on_entry(self, entry):
        with (self.directory / "live_trace.jsonl").open("a") as output:
            output.write(json.dumps(entry.to_dict(), default=str) + "\n")
        if entry.event_type == "tool_call":
            fingerprint = hashlib.sha256(json.dumps(
                [entry.tool_name, entry.tool_args], sort_keys=True, default=str
            ).encode()).hexdigest()
            self.pending[entry.tool_name].append(fingerprint)
            self.repeats[fingerprint] += 1
            self.tools += 1
            self.emit("tool_call", tool=entry.tool_name, number=self.tools,
                      identical_calls=self.repeats[fingerprint])
        elif entry.event_type == "tool_response":
            queue = self.pending[entry.tool_name]
            fingerprint = queue.popleft() if queue else entry.tool_name
            try:
                payload = json.loads(entry.tool_result)
            except (ValueError, TypeError):
                payload = {}
            if not isinstance(payload, dict):
                payload = {}
            details = payload.get("result", payload)
            exit_code = details.get("exit_code") if isinstance(details, dict) else None
            failed = payload.get("status") == "error" or bool(payload.get("error")) or exit_code not in (None, 0)
            self.emit("tool_response", tool=entry.tool_name, status="error" if failed else payload.get("status", "unknown"),
                      error_type=payload.get("error_type"), exit_code=exit_code)
            if failed:
                self.failures[fingerprint] += 1
                if self.failures[fingerprint] >= 2:
                    self.pause("The same tool call failed twice; review before continuing.")
            self.nudges = 0
            if self.repeats[fingerprint] >= 4:
                self.pause("An identical tool call was repeated four times; review for a loop.")
        elif entry.event_type == "continuation_nudge":
            self.nudges += 1
            self.emit("continuation_nudge", consecutive=self.nudges)
            if self.nudges >= 2:
                self.pause("Two text-only endings without submit_patch; review tool use.")
        elif entry.event_type in {"compaction", "error"}:
            self.emit(entry.event_type)

    async def inspect_workspace(self):
        if not self.ctx.container_id:
            return
        status = await self.ctx.exec_async("cd /workspace && git status --porcelain", timeout=15)
        diff = await self.ctx.exec_async("cd /workspace && git diff --binary _swegemma_baseline", timeout=15)
        if status.exit_code or diff.exit_code:
            self.pause("The workspace diff could not be inspected.")
            return
        (self.directory / "workspace-status.txt").write_text(status.stdout)
        (self.directory / "workspace.diff").write_text(diff.stdout)
        self.changed_files = len(status.stdout.splitlines())
        digest = hashlib.sha256((status.stdout + diff.stdout).encode()).hexdigest()
        if digest != self.diff_hash:
            self.diff_hash = digest
            self.last_progress = self.tools
            self.repeats.clear()
            self.emit("workspace", changed_files=self.changed_files, patch_bytes=len(diff.stdout.encode()))
        if self.tools - self.last_progress >= 12 and not self.changed_files:
            self.pause("Twelve tool calls with no workspace edits; review progress.")

    async def before_model(self, request):
        await self.inspect_workspace()
        # Upstream retains a 500-call fallback when turns=None. Review before it.
        if self.calls == 450:
            self.pause("Approaching the upstream 500-call fallback; review before continuing.")
        pause_path = self.directory / "pause.json"
        if pause_path.exists():
            self.paused = True
            self.emit("paused", control_file=str(pause_path))
            while pause_path.exists():
                await asyncio.sleep(1)
            self.paused = False
            self.failures.clear()
            self.repeats.clear()
            self.nudges = 0
            self.last_progress = self.tools
            self.emit("resumed")
        steering = self.directory / "steering.txt"
        if steering.exists():
            from google.genai import types
            text = steering.read_text()
            request.contents.append(types.Content(role="user", parts=[types.Part(text=text)]))
            with (self.directory / "steering-history.jsonl").open("a") as history:
                history.write(json.dumps({"model_call": self.calls + 1, "text": text}) + "\n")
            steering.unlink()
            self.emit("user_guidance_applied")
        self.calls += 1
        self.emit("model_request", number=self.calls, changed_files=self.changed_files)

    async def heartbeat(self):
        while True:
            await asyncio.sleep(30)
            self.emit("heartbeat", phase=self.ctx.current_phase, model_calls=self.calls,
                      tools=self.tools, changed_files=self.changed_files, paused=self.paused)


async def evaluate(args, directory):
    from adk_eval_core.tracing import SessionTrace
    from swegemma.budget import EvaluationBudget
    from swegemma.cli import _build_cli_model_registry
    from swegemma.config import EvalConfig, EventsCompactionConfig
    from swegemma.context import SwegemmaContext
    from swegemma.evaluate import Evaluator
    import swegemma.harness.agent_runner as runner
    from swegemma.models import load_tasks

    tasks = load_tasks(DATA / "tasks.jsonl")
    task = next(t for t in tasks if t.instance_id == args.task_id) if args.task_id else tasks[0]
    config = EvalConfig(
        tasks_path=DATA / "tasks.jsonl", snapshots_dir=DATA / "snapshots",
        results_dir=directory, submission_dir=ROOT / "submission",
        models=_build_cli_model_registry(ROOT / "submission", args.models_yaml),
        image="localhost/gemma-swebench-sandbox-local:2026-10-14", sandbox="docker",
        wheels_dir=DATA / "wheels", graph_dir=str(DATA / "graphs"), embeddings_dir=str(DATA / "embeddings"),
        budget=EvaluationBudget(time_minutes=args.minutes, tool_calls=None, turns=None),
        events_compaction_config=EventsCompactionConfig(
            compaction_interval=15, overlap_size=2, token_threshold=14336, event_retention_size=5,
        ),
        enable_sandbox_testing=False, display_mode="quiet", skip_agent_patch=args.baseline,
    )
    evaluator = Evaluator(config)
    ctx = SwegemmaContext(docker_manager=evaluator.sandbox, task=task, repo=task.repo,
                         budget=config.budget, harness=config.harness,
                         graph_dir=config.graph_dir, embeddings_dir=config.embeddings_dir)
    monitor = Monitor(directory, ctx)
    original_trace, original_plugin = runner.SessionTrace, runner.EventDisplayPlugin

    class LiveTrace(SessionTrace):
        def __init__(self):
            super().__init__()
            self.set_callback(monitor.on_entry)

    class SupervisedDisplay(original_plugin):
        async def before_model_callback(self, *, callback_context, llm_request):
            await monitor.before_model(llm_request)
            return None

    runner.SessionTrace, runner.EventDisplayPlugin = LiveTrace, SupervisedDisplay
    heartbeat = asyncio.create_task(monitor.heartbeat())
    monitor.emit("started", task_id=task.instance_id, baseline=args.baseline,
                 time_limit_minutes=args.minutes, turn_limit=None, tool_limit=None,
                 upstream_call_fallback=500, results=str(directory))
    try:
        result = await evaluator.evaluate_task(task, context=ctx)
        (directory / "agent.patch").write_text(result.agent_patch)
        (directory / "test_output.log").write_text(result.test_output)
        save_json(directory / "result.json", result.model_dump(exclude={"trace"}))
        summary = {"task_id": task.instance_id, "resolved": result.resolved,
                   "test_exit_code": result.test_exit_code, "error": result.error,
                   "duration_seconds": result.duration_seconds,
                   "model_calls": result.total_llm_calls, "tools": result.tool_calls,
                   "patch_bytes": len(result.agent_patch.encode()),
                   "test_counts": re.findall(r"\b\d+ (?:passed|failed|errors?|skipped|xfailed)\b", result.test_output)}
        save_json(directory / "summary.json", summary)
        monitor.emit("completed", **summary)
        return 1 if result.error else 0
    except asyncio.CancelledError:
        save_json(directory / "summary.json", {
            "status": "interrupted", "task_id": task.instance_id,
            "model_requests_started": monitor.calls, "tools": monitor.tools,
            "changed_files": monitor.changed_files,
        })
        monitor.emit("interrupted", model_requests_started=monitor.calls, tools=monitor.tools)
        raise
    finally:
        heartbeat.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await heartbeat
        runner.SessionTrace, runner.EventDisplayPlugin = original_trace, original_plugin
        evaluator.sandbox.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", help="Defaults to the first dataset entry")
    parser.add_argument("--baseline", action="store_true", help="Verify unpatched code without a model")
    parser.add_argument("--minutes", type=float, default=None, help="Optional explicit session time limit")
    parser.add_argument("--models-yaml", type=Path, default=ROOT / "dev/models.vllm.yaml")
    args = parser.parse_args()
    mapping = yaml.safe_load(args.models_yaml.read_text())
    declared = yaml.safe_load((ROOT / "submission/agent.yaml").read_text())["model"]
    if declared not in mapping["models"]:
        parser.error("The submission model must have an explicit local endpoint mapping")
    endpoint = mapping["models"][declared]["api_base"]
    for value in mapping["models"].values():
        if urlparse(value["api_base"]).hostname not in {"127.0.0.1", "localhost", "::1"}:
            parser.error("This runner only permits a local model endpoint")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory = ROOT / "results" / (stamp + ("-unpatched" if args.baseline else "-vllm-metal-12b"))
    directory.mkdir(parents=True)
    for path in (ROOT / "submission/agent.yaml", args.models_yaml, ROOT / "dev/vllm-runtime.json"):
        (directory / path.name).write_text(path.read_text())
    save_json(directory / "run.json", {"runtime": "local-vllm-metal-arm64", "arguments": vars(args),
                                      "budget": {"time_minutes": args.minutes, "turns": None, "tool_calls": None},
                                      "upstream_call_fallback": 500})
    os.environ.update({"KAGGLE_SANDBOX_DIR": str(DATA / "sandbox"),
                       "LITELLM_LOCAL_MODEL_COST_MAP": "True", "LITELLM_TELEMETRY": "False",
                       "OTEL_SDK_DISABLED": "true", "HF_HUB_OFFLINE": "1",
                       "MODEL_PROXY_URL": endpoint, "MODEL_PROXY_API_KEY": "local-only"})
    print(f"Local result directory: {directory}", flush=True)
    with (directory / "harness.log").open("w") as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            try:
                return asyncio.run(evaluate(args, directory))
            except KeyboardInterrupt:
                return 130


if __name__ == "__main__":
    raise SystemExit(main())
