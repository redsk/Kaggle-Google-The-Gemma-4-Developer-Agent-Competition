#!/usr/bin/env python3
"""Pinned, workspace-local CLIProxyAPI setup and lifecycle. No task execution."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import secrets
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / ".cache/cliproxyapi"
CONFIG = STATE / "config.yaml"
MANIFEST = json.loads((ROOT / "dev/cliproxyapi-runtime.json").read_text())
INSTALL = ROOT / "downloads/tools" / f"cliproxyapi-{MANIFEST['version']}"
BINARY = INSTALL / "cli-proxy-api"


def setup():
    if (platform.system(), platform.machine()) != ("Darwin", "arm64"):
        raise SystemExit("This pinned binary is for Apple Silicon macOS.")
    INSTALL.mkdir(parents=True, exist_ok=True)
    archive = INSTALL / MANIFEST["archive_url"].rsplit("/", 1)[1]
    if not archive.exists():
        print(f"Downloading CLIProxyAPI {MANIFEST['version']}...", flush=True)
        temporary = archive.with_suffix(".download")
        try:
            with urllib.request.urlopen(MANIFEST["archive_url"], timeout=60) as response:
                temporary.write_bytes(response.read())
            temporary.replace(archive)
        finally:
            temporary.unlink(missing_ok=True)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != MANIFEST["archive_sha256"]:
        raise SystemExit("Archive checksum mismatch; no files installed.")
    # Extract only known regular files; do not trust archive paths or links.
    with tarfile.open(archive, "r:gz") as bundle:
        for name in ("cli-proxy-api", "LICENSE", "config.example.yaml"):
            matches = [m for m in bundle.getmembers() if Path(m.name).name == name and m.isfile()]
            if len(matches) != 1:
                raise SystemExit(f"Expected exactly one regular {name} in archive.")
            with bundle.extractfile(matches[0]) as source:
                (INSTALL / name).write_bytes(source.read())
    BINARY.chmod(0o755)
    STATE.mkdir(parents=True, exist_ok=True, mode=0o700)
    STATE.chmod(0o700)
    auth = STATE / "auth"
    auth.mkdir(exist_ok=True, mode=0o700)
    auth.chmod(0o700)
    if not CONFIG.exists():
        template = (ROOT / "dev/cliproxyapi.example.yaml").read_text()
        content = template.replace("__AUTH_DIR__", json.dumps(str(auth)))
        content = content.replace("__LOCAL_API_KEY__", json.dumps(secrets.token_urlsafe(32)))
        # umask is private before the first byte is written.
        CONFIG.write_text(content)
    CONFIG.chmod(0o600)
    print(f"Ready: CLIProxyAPI {MANIFEST['version']}; private config: {CONFIG}")


def client_settings():
    import yaml

    if not CONFIG.exists():
        raise SystemExit("Run .venv/bin/python scripts/cliproxy.py setup first.")
    config = yaml.safe_load(CONFIG.read_text())
    host = config["server"]["host"]
    if host != "127.0.0.1":
        raise SystemExit("This experiment expects a loopback-only proxy.")
    return f"http://{host}:{config['server']['port']}/v1", config["access"]["api-keys"][0]


def model_catalog():
    endpoint, key = client_settings()
    request = urllib.request.Request(endpoint + "/models", headers={"Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(request, timeout=15) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("setup", "login", "serve", "models"))
    args = parser.parse_args()
    os.umask(0o077)
    if args.command == "setup":
        setup()
        return
    client_settings()  # Validate config before launching or making requests.
    if args.command == "models":
        for entry in model_catalog().get("data", []):
            print(entry["id"])
        print("Advertised models only; a successful request must verify access.")
        return
    if not BINARY.exists():
        raise SystemExit("Run .venv/bin/python scripts/cliproxy.py setup first.")
    # Keep logs and credentials in ignored local state; use the release's catalog.
    os.chdir(STATE)
    command = [str(BINARY), "-config", str(CONFIG), "-local-model"]
    if args.command == "login":
        command.append("-codex-login")
    os.execv(BINARY, command)


if __name__ == "__main__":
    main()
