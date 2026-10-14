"""Fetch official harness assets and one task without printing task contents."""

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

from kaggle.api.kaggle_api_extended import KaggleApi

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "downloads" / "competition"
COMPETITION = "gemma-4-developer-agent"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", help="Public task to fetch; defaults to the first dataset entry")
    args = parser.parse_args()
    api = KaggleApi()
    api.authenticate()
    DATA.mkdir(parents=True, exist_ok=True)
    files = []
    token = None
    while True:
        page = api.competition_list_files(COMPETITION, page_token=token, page_size=200)
        files.extend({"name": f.name, "size": f.total_bytes} for f in page.files)
        token = page.next_page_token
        if not token:
            break
    (DATA / "files_manifest.json").write_text(json.dumps(files, indent=2) + "\n")
    available = {f["name"]: f for f in files}

    def fetch(name):
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid download path")
        dest = DATA / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.is_file() and dest.stat().st_size == available[name]["size"]:
            return
        api.competition_download_file(COMPETITION, name, path=str(dest.parent), quiet=True)
        archive = dest.with_name(dest.name + ".zip")
        if not dest.exists() and archive.exists():
            with ZipFile(archive) as zipped:
                matches = [n for n in zipped.namelist() if Path(n).name == dest.name]
                if len(matches) != 1:
                    raise ValueError(f"Unexpected download archive for {name}")
                dest.write_bytes(zipped.read(matches[0]))
        if not dest.is_file() or dest.stat().st_size != available[name]["size"]:
            raise ValueError(f"Download size mismatch: {name}")
        print(f"Downloaded {name}", flush=True)

    fetch("tasks.jsonl")
    tasks = [json.loads(line) for line in (DATA / "tasks.jsonl").read_text().splitlines() if line]
    task = next((task for task in tasks if task["instance_id"] == args.task_id), None) if args.task_id else tasks[0]
    if task is None:
        raise ValueError(f"Unknown public task: {args.task_id}")
    selected = [f["name"] for f in files if f["name"] == "HARNESS_README.md" or f["name"].startswith(("docker/", "sandbox/", "wheels/", "sample_submission/"))]
    selected.append(f"snapshots/{task['instance_id']}.tgz")
    repo = task["repo"].rsplit("/", 1)[-1]
    for directory, suffix in (("graphs", "json"), ("embeddings", "npz")):
        name = f"{directory}/{repo}_{task['base_commit']}.{suffix}"
        if name in available:
            selected.append(name)
    for name in selected:
        fetch(name)
    smoke = {"competition": COMPETITION, "task_id": task["instance_id"], "repo": task["repo"], "files": {}}
    for name in ["tasks.jsonl", *selected]:
        smoke["files"][name] = hashlib.sha256((DATA / name).read_bytes()).hexdigest()
    (DATA / "smoke_manifest.json").write_text(json.dumps(smoke, indent=2) + "\n")
    print(f"Smoke assets ready: {task['instance_id']}; {len(selected)} supporting files", flush=True)


if __name__ == "__main__":
    main()
