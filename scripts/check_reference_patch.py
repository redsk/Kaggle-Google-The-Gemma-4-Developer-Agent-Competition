"""Verify the published reference patch locally; never expose it to the agent."""

import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "downloads/competition"
os.environ.update({
    "KAGGLE_SANDBOX_DIR": str(DATA / "sandbox"),
    "LITELLM_LOCAL_MODEL_COST_MAP": "True",
    "LITELLM_TELEMETRY": "False",
    "OTEL_SDK_DISABLED": "true",
    "OTEL_PYTHON_DISABLED": "true",
    "HF_HUB_OFFLINE": "1",
})

from adk_submission import ModelRegistry
from swegemma.config import EvalConfig
from swegemma.harness.verification import verify_task
from swegemma.models import load_tasks
from swegemma.sandbox import ContainerConfig, ContainerManager


async def main():
    manifest = json.loads((DATA / "smoke_manifest.json").read_text())
    task = next(t for t in load_tasks(DATA / "tasks.jsonl") if t.instance_id == manifest["task_id"])
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    results = ROOT / "results" / f"{timestamp}-reference"
    results.mkdir(parents=True)
    config = EvalConfig(
        tasks_path=DATA / "tasks.jsonl", snapshots_dir=DATA / "snapshots",
        results_dir=results, submission_dir=ROOT / "submission", models=ModelRegistry(),
        image="localhost/gemma-swebench-sandbox-local:2026-10-14",
        verification_timeout_seconds=180,
    )
    manager = ContainerManager(ContainerConfig(image=config.image))
    try:
        result = await verify_task(
            manager, config, task, DATA / "snapshots" / f"{task.instance_id}.tgz",
            agent_patch=task.patch, start_time=time.perf_counter(),
        )
    finally:
        manager.close()
    summary = {"mode": "reference-patch-fixture-check", "task_id": task.instance_id,
               "runtime": "local-arm64-with-workspace-import-priority",
               "resolved": result.resolved, "test_exit_code": result.test_exit_code,
               "error": result.error, "duration_seconds": result.duration_seconds}
    (results / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (results / "test_output.log").write_text(result.test_output)
    print(json.dumps(summary, indent=2))
    print(f"Results: {results}")
    return 0 if result.resolved and result.error is None else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
