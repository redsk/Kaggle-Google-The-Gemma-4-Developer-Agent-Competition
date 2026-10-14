"""Test a saved model-generated patch locally without making any model calls."""

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
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
import importlib
from swegemma.models import load_tasks
from swegemma.sandbox import ContainerConfig, ContainerManager


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("patch", type=Path)
    parser.add_argument("--task-id")
    args = parser.parse_args()
    tasks = load_tasks(DATA / "tasks.jsonl")
    run_file = args.patch.parent / "summary.json"
    recorded_task = json.loads(run_file.read_text()).get("task_id") if run_file.exists() else None
    task_id = args.task_id or recorded_task or tasks[0].instance_id
    task = next(t for t in tasks if t.instance_id == task_id)
    patch = args.patch.read_text()
    if not patch.strip():
        raise ValueError("The patch file is empty.")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    results = ROOT / "results" / f"{stamp}-saved-patch-verification"
    os.umask(0o077)
    results.mkdir(parents=True, mode=0o700)
    config = EvalConfig(
        tasks_path=DATA / "tasks.jsonl", snapshots_dir=DATA / "snapshots",
        results_dir=results, submission_dir=ROOT / "submission", models=ModelRegistry(),
        image="localhost/gemma-swebench-sandbox-local:2026-10-14",
    )
    manager = ContainerManager(ContainerConfig(image=config.image))
    from observability import Telemetry, instrument
    telemetry = Telemetry(results, metadata={"task_id": task_id, "source_run": args.patch.parent.name,
                          "scope": "saved patch verification; no model calls"})
    try:
        with telemetry.run("verification.saved"), instrument():
            result = await importlib.import_module("swegemma.evaluate").verify_task(
                manager, config, task, DATA / "snapshots" / f"{task.instance_id}.tgz",
                agent_patch=patch, start_time=time.perf_counter(),
            )
    finally:
        manager.close()
    counts = re.findall(r"\b\d+ (?:passed|failed|errors?|skipped|xfailed|xpassed)\b", result.test_output)
    summary = {
        "mode": "saved-patch-verification-only", "task_id": task_id,
        "patch_file": str(args.patch.resolve()),
        "patch_sha256": hashlib.sha256(patch.encode()).hexdigest(),
        "runtime": "local-arm64-with-workspace-import-priority",
        "resolved": result.resolved, "test_exit_code": result.test_exit_code,
        "error": result.error, "duration_seconds": result.duration_seconds,
        "test_counts": counts,
    }
    (results / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (results / "test_output.log").write_text(result.test_output)
    print(json.dumps(summary, indent=2))
    print(f"Results: {results}")
    return 0 if result.resolved and result.error is None else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
