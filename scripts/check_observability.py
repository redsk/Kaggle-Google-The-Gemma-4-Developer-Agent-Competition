#!/usr/bin/env python3
"""Send scripted, synthetic traces to local Phoenix. No inference or containers."""
from datetime import datetime, timezone
import asyncio
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
os.environ.update({"LITELLM_LOCAL_MODEL_COST_MAP": "True", "LITELLM_TELEMETRY": "False", "OTEL_SDK_DISABLED": "true"})
from observability_fixture import exercise
from observability import DEFAULT_ENDPOINT

if __name__ == "__main__":
    os.umask(0o077)
    directory = ROOT / "results" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-observability-preflight")
    result = asyncio.run(exercise(directory, endpoint=DEFAULT_ENDPOINT))
    (directory / "checks.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
