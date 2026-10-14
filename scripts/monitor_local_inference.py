#!/usr/bin/env python3
"""Run a local command while recording macOS inference diagnostics (no sudo).

Example:
  .venv/bin/python scripts/monitor_local_inference.py --label vllm -- \
    .venv/bin/python scripts/benchmark_12b.py vllm --start-server

Records system-wide VM counters, GPU utilization/allocation, thermal pressure,
power mode, and process CPU/RSS. It does not change system or model settings.
Pass --macmon PATH to also record GPU clocks, temperatures, fans and power.
GPU utilization covers every process; RSS and swap allocation alone do not
establish memory pressure. Compare counter deltas with the generation events.
"""

import argparse
import ctypes
from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import plistlib
import re
import signal
import subprocess
import sys
import threading
import time


ROOT = Path(__file__).resolve().parents[1]
THERMAL_NAMES = {0: "nominal", 1: "fair", 2: "serious", 3: "critical"}


def command_output(command):
    result = subprocess.run(command, capture_output=True, timeout=5)
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors="replace").strip())
    return result.stdout


class PowerState:
    def __init__(self):
        self.foundation = ctypes.CDLL(
            "/System/Library/Frameworks/Foundation.framework/Foundation")
        self.objc = ctypes.CDLL("/usr/lib/libobjc.A.dylib")
        self.objc.objc_getClass.argtypes = [ctypes.c_char_p]
        self.objc.objc_getClass.restype = ctypes.c_void_p
        self.objc.sel_registerName.argtypes = [ctypes.c_char_p]
        self.objc.sel_registerName.restype = ctypes.c_void_p
        address = ctypes.cast(self.objc.objc_msgSend, ctypes.c_void_p).value
        send = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)(address)
        self.number = ctypes.CFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p)(address)
        self.process_info = send(self.objc.objc_getClass(b"NSProcessInfo"),
                                 self.objc.sel_registerName(b"processInfo"))

    def read(self):
        thermal = self.number(self.process_info, self.objc.sel_registerName(b"thermalState"))
        low = self.number(self.process_info, self.objc.sel_registerName(b"isLowPowerModeEnabled"))
        return {"thermal_state": thermal,
                "thermal_state_name": THERMAL_NAMES.get(thermal, "unknown"),
                "low_power_mode": bool(low)}


def snapshot(power):
    result = {"utc": datetime.now(timezone.utc).isoformat(), "monotonic_s": time.monotonic()}
    readers = {
        "power": power.read,
        "vm": read_vm,
        "gpu": read_gpu,
        "processes": read_processes,
        "swap": lambda: command_output(["/usr/sbin/sysctl", "-n", "vm.swapusage"]).decode().strip(),
        "memory_pressure_level": lambda: int(command_output(
            ["/usr/sbin/sysctl", "-n", "kern.memorystatus_vm_pressure_level"])),
    }
    for name, reader in readers.items():
        try:
            result[name] = reader()
        except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as exc:
            result[name + "_error"] = str(exc)
    return result


def read_vm():
    raw = command_output(["/usr/bin/vm_stat"]).decode()
    values = {m[0].strip(): int(m[1]) for m in re.findall(r"^([^:]+):\s*(\d+)\.", raw, re.M)}
    values["page_size"] = int(re.search(r"page size of (\d+) bytes", raw)[1])
    return values


def read_gpu():
    nodes = plistlib.loads(command_output(["/usr/sbin/ioreg", "-r", "-c", "AGXAccelerator", "-a"]))
    return [n.get("PerformanceStatistics", {}) for n in nodes]


def read_processes():
    raw = command_output(["/bin/ps", "-axo", "pid,ppid,%cpu,rss,comm", "-r"]).decode()
    rows = []
    for line in raw.splitlines()[1:]:
        parts = line.split(None, 4)
        if len(parts) == 5:
            rows.append({"pid": int(parts[0]), "ppid": int(parts[1]),
                         "cpu_percent": float(parts[2]), "rss_kib": int(parts[3]),
                         "executable": parts[4]})
    inference = [r for r in rows if any(s in r["executable"].lower()
                                      for s in ("python", "vllm", "ollama"))]
    return {"top_cpu": rows[:12], "inference": inference}


@contextmanager
def macmon_capture(directory, binary, interval):
    if binary is None:
        yield None
        return
    with (directory / "macmon.jsonl").open("w") as output, \
            (directory / "macmon.stderr.log").open("w") as errors:
        process = subprocess.Popen(
            [str(binary), "pipe", "-i", str(max(1, round(interval * 1000)))],
            stdout=output, stderr=errors, start_new_session=True)
        try:
            yield process
        finally:
            stopped_by_monitor = process.poll() is None
            if stopped_by_monitor:
                process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            (directory / "macmon-exit.json").write_text(json.dumps({
                "returncode": process.returncode,
                "stopped_by_monitor": stopped_by_monitor}) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="inference")
    parser.add_argument("--interval", type=float, default=2)
    parser.add_argument("--macmon", type=Path,
                        help="Optional macmon executable; records read-only hardware sensors.")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cmd = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not cmd or args.interval <= 0:
        parser.error("Supply a command after -- and a positive interval.")
    if args.macmon is not None:
        args.macmon = args.macmon.resolve()
        if not args.macmon.is_file():
            parser.error(f"macmon executable not found: {args.macmon}")
    label = re.sub(r"[^a-zA-Z0-9_-]", "-", args.label)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory = ROOT / "results" / f"{stamp}-monitor-{label}"
    directory.mkdir()
    print(f"Diagnostics: {directory}", flush=True)
    power = PowerState()
    metadata = {"command": cmd, "interval_seconds": args.interval,
                "macmon": str(args.macmon) if args.macmon else None,
                "notes": ["GPU counters are system-wide, not per-process.",
                          "Without --macmon, GPU frequency and temperature are not recorded."]}
    for name, command in [("power_settings", ["/usr/bin/pmset", "-g", "custom"]),
                          ("battery", ["/usr/bin/pmset", "-g", "batt"])]:
        try:
            metadata[name] = command_output(command).decode()
        except Exception as exc:
            metadata[name + "_error"] = str(exc)
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    stage = {"name": "starting"}
    with macmon_capture(directory, args.macmon, args.interval) as sensor, \
            (directory / "console.jsonl").open("w") as console, \
            (directory / "telemetry.jsonl").open("w") as telemetry:
        telemetry.write(json.dumps(snapshot(power)) + "\n")
        telemetry.flush()
        child = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, bufsize=1,
                                 start_new_session=True)
        # Prevent idle system sleep only while this diagnostic command runs.
        awake = subprocess.Popen(["/usr/bin/caffeinate", "-i", "-w", str(child.pid)])

        def copy_output():
            for line in child.stdout:
                if line.startswith(("Run ", "Warming", "Starting", "Median")):
                    stage["name"] = line.strip()
                console.write(json.dumps({"utc": datetime.now(timezone.utc).isoformat(),
                                          "monotonic_s": time.monotonic(), "line": line.rstrip()}) + "\n")
                console.flush()
                print(line, end="", flush=True)

        output_thread = threading.Thread(target=copy_output, daemon=True)
        output_thread.start()
        previous_thermal = None
        sensor_failure_reported = False
        interrupted = False
        try:
            while child.poll() is None:
                if sensor is not None and sensor.poll() is not None and not sensor_failure_reported:
                    print("[monitor] macmon exited early; see macmon.stderr.log", flush=True)
                    sensor_failure_reported = True
                row = snapshot(power)
                row["stage"] = stage["name"]
                telemetry.write(json.dumps(row) + "\n")
                telemetry.flush()
                thermal = row.get("power", {}).get("thermal_state_name")
                if thermal != previous_thermal:
                    print(f"[monitor] thermal pressure: {thermal}", flush=True)
                    previous_thermal = thermal
                time.sleep(args.interval)
        except KeyboardInterrupt:
            interrupted = True
            import os
            os.killpg(child.pid, signal.SIGINT)
            child.wait()  # Let the benchmark run its server cleanup.
        finally:
            if child.poll() is None:
                import os
                os.killpg(child.pid, signal.SIGINT)
                child.wait()
            output_thread.join(timeout=5)
            telemetry.write(json.dumps(snapshot(power)) + "\n")
            awake.terminate()
            awake.wait()
        (directory / "exit.json").write_text(json.dumps(
            {"returncode": child.returncode, "interrupted": interrupted}) + "\n")
    print(f"Saved diagnostics: {directory}", flush=True)
    return child.returncode


if __name__ == "__main__":
    sys.exit(main())
