#!/usr/bin/env python3
"""Benchmark local 12B chat inference; no competition tasks or cloud calls.

Examples:
  .venv/bin/python scripts/benchmark_12b.py vllm --start-server
  .venv/bin/python scripts/benchmark_12b.py ollama --start-server
  .venv/bin/python scripts/benchmark_12b.py vllm --prompt 'Explain backpropagation.'

Uses only Python's standard library. Servers started by this script are stopped
on exit. An already-running server is reused and left running.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import sys
import time
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener


ROOT = Path(__file__).resolve().parents[1]
URLS = {"ollama": "http://127.0.0.1:11434", "vllm": "http://127.0.0.1:8000"}
MODELS = {"ollama": "gemma4-12b-chat", "vllm": "gemma4-12b-local"}
DEFAULT_PROMPT = (
    "Explain how a neural network learns, for a programmer who knows Python "
    "but is new to machine learning. Cover weights, biases, activations, the "
    "forward pass, the loss function, backpropagation, gradient descent, and "
    "the difference between training and inference. Include a small numerical "
    "example and explain how the concepts connect. Aim for about 800 words."
)
HTTP = build_opener(ProxyHandler({}))  # These requests must stay on localhost.


def request(url, body=None, timeout=15):
    data = None if body is None else json.dumps(body).encode()
    req = Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        return HTTP.open(req, timeout=timeout)
    except HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:1500]
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from exc


def json_request(url, body=None):
    with request(url, body) as response:
        return json.load(response)


def ready(engine):
    try:
        path = "/api/version" if engine == "ollama" else "/health"
        with request(URLS[engine] + path, timeout=2):
            return True
    except (OSError, RuntimeError):
        return False


def ollama_loaded():
    if not ready("ollama"):
        return []
    return json_request(URLS["ollama"] + "/api/ps").get("models", [])


def metrics():
    """Read cumulative vLLM metrics; deltas are valid only for an isolated call."""
    try:
        with request(URLS["vllm"] + "/metrics", timeout=5) as response:
            raw = response.read().decode()
    except (OSError, RuntimeError):
        return {}
    values = {}
    for line in raw.splitlines():
        match = re.match(r"^(vllm:[a-zA-Z0-9_:]+)(?:\{[^}]*\})?\s+([\d.eE+\-]+)$", line)
        if match:
            key, value = match.groups()
            values[key] = values.get(key, 0.0) + float(value)
    return values


def metric_delta(before, after, key):
    name = "vllm:" + key
    if name not in before or name not in after:
        return None
    value = after[name] - before[name]
    return value if value >= 0 else None


def rate(count, seconds):
    return count / seconds if count is not None and seconds and seconds > 0 else None


def make_payload(engine, messages, args, maximum):
    common = {"model": MODELS[engine], "messages": messages, "stream": True}
    if engine == "ollama":
        return {**common, "think": args.think, "keep_alive": "10m", "options": {
            "num_ctx": args.context, "num_predict": maximum,
            "temperature": args.temperature, "top_p": 0.95, "top_k": 64, "seed": 42,
        }}
    return {**common, "max_completion_tokens": maximum,
            "temperature": args.temperature, "top_p": 0.95, "top_k": 64, "seed": 42,
            "chat_template_kwargs": {"enable_thinking": args.think},
            "stream_options": {"include_usage": True}}


def generate(engine, messages, args, directory, label, maximum):
    payload = make_payload(engine, messages, args, maximum)
    (directory / f"{label}-request.json").write_text(json.dumps(payload, indent=2) + "\n")
    before = metrics() if engine == "vllm" else {}
    path = "/api/chat" if engine == "ollama" else "/v1/chat/completions"
    first = last = first_answer = None
    answer, thinking = [], []
    usage, final, finish_reason = {}, {}, None
    complete = False
    started = time.perf_counter()
    with (directory / f"{label}-events.jsonl").open("w") as events:
        with request(URLS[engine] + path, payload, timeout=1800) as response:
            for raw in response:
                line = raw.decode().strip()
                if not line or (engine == "vllm" and not line.startswith("data:")):
                    continue
                line = line[5:].strip() if engine == "vllm" else line
                if line == "[DONE]":
                    complete = True
                    break
                item = json.loads(line)
                elapsed = time.perf_counter() - started
                events.write(json.dumps({"elapsed_s": elapsed, "response": item}) + "\n")
                events.flush()
                if item.get("error"):
                    raise RuntimeError(str(item["error"]))
                if engine == "ollama":
                    delta = item.get("message", {})
                    text, thought = delta.get("content", ""), delta.get("thinking", "")
                    if item.get("done"):
                        final, complete = item, True
                        finish_reason = item.get("done_reason")
                else:
                    if item.get("usage"):
                        usage = item["usage"]
                    choices = item.get("choices", [])
                    delta = choices[0].get("delta", {}) if choices else {}
                    text = delta.get("content") or ""
                    thought = delta.get("reasoning_content") or delta.get("reasoning") or ""
                    if choices and choices[0].get("finish_reason"):
                        finish_reason = choices[0]["finish_reason"]
                if text or thought:
                    first = elapsed if first is None else first
                    last = elapsed
                if text:
                    first_answer = elapsed if first_answer is None else first_answer
                    answer.append(text)
                    if args.show_output and label != "warmup":
                        print(text, end="", flush=True)
                if thought:
                    thinking.append(thought)
                if engine == "ollama" and complete:
                    break
    wall = time.perf_counter() - started
    if args.show_output and label != "warmup":
        print()
    if not complete:
        raise RuntimeError("The response stream ended without a completion marker.")
    (directory / f"{label}-answer.txt").write_text("".join(answer))
    if thinking:
        (directory / f"{label}-thinking.txt").write_text("".join(thinking))
    result = {"wall_seconds": wall, "time_to_first_output_seconds": first,
              "time_to_first_answer_seconds": first_answer, "finish_reason": finish_reason,
              "thinking_characters": sum(map(len, thinking)), "warnings": []}
    if engine == "ollama":
        result.update({"prompt_tokens": final.get("prompt_eval_count"),
                       "cached_prompt_tokens": final.get("prompt_eval_cached_count"),
                       "completion_tokens": final.get("eval_count"),
                       "prompt_seconds": final.get("prompt_eval_duration", 0) / 1e9,
                       "decode_seconds": final.get("eval_duration", 0) / 1e9,
                       "load_seconds": final.get("load_duration", 0) / 1e9,
                       "server_total_seconds": final.get("total_duration", 0) / 1e9})
        # Older Ollama versions do not expose cache hits; don't invent a prefill rate.
        if result["cached_prompt_tokens"] is not None and result["prompt_tokens"] is not None:
            result["uncached_prompt_tokens"] = result["prompt_tokens"] - result["cached_prompt_tokens"]
        result["generation_tokens_per_second"] = rate(result["completion_tokens"], result["decode_seconds"])
        result["generation_rate_source"] = "Ollama eval_count / eval_duration"
    else:
        if "completion_tokens" not in usage:
            raise RuntimeError("vLLM did not return token usage; cannot compute tokens/second.")
        result.update({"prompt_tokens": usage.get("prompt_tokens"),
                       "completion_tokens": usage["completion_tokens"], "usage": usage})
        after = metrics()
        result["server_metric_deltas"] = {
            key: metric_delta(before, after, key) for key in (
                "request_success_total", "request_generation_tokens_sum",
                "request_prefill_time_seconds_sum", "request_decode_time_seconds_sum",
                "request_prefill_kv_computed_tokens_sum", "e2e_request_latency_seconds_sum")}
        d = result["server_metric_deltas"]
        if d["request_success_total"] == 1 and d["request_generation_tokens_sum"] == usage["completion_tokens"]:
            result["prompt_seconds"] = d["request_prefill_time_seconds_sum"]
            result["decode_seconds"] = d["request_decode_time_seconds_sum"]
            result["uncached_prompt_tokens"] = d["request_prefill_kv_computed_tokens_sum"]
            result["server_total_seconds"] = d["e2e_request_latency_seconds_sum"]
            # The first generated token belongs to prefill, not the decode phase.
            result["generation_tokens_per_second"] = rate(max(0, usage["completion_tokens"] - 1), result["decode_seconds"])
            result["generation_rate_source"] = "vLLM (completion_tokens - 1) / measured decode time"
        else:
            result["warnings"].append("vLLM metrics missing, delayed, or affected by another request; using stream estimate.")
    count = result.get("completion_tokens")
    if count is None or count <= 0:
        raise RuntimeError("The server returned no generated-token count.")
    stream_seconds = last - first if first is not None and last is not None else None
    result["stream_estimated_tokens_per_second"] = rate(max(0, count - 1), stream_seconds)
    if result.get("generation_tokens_per_second") is None:
        result["generation_tokens_per_second"] = result["stream_estimated_tokens_per_second"]
        result["generation_rate_source"] = "client stream estimate (chunks can contain multiple tokens)"
    result["uncached_prompt_tokens_per_second"] = rate(result.get("uncached_prompt_tokens"), result.get("prompt_seconds"))
    result["end_to_end_tokens_per_second"] = rate(count, wall)
    if not args.think and thinking:
        result["warnings"].append("The server emitted thinking despite thinking being disabled.")
    (directory / f"{label}-metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def show(result):
    fields = [("total wall time", "wall_seconds", "s"),
              ("time to first output", "time_to_first_output_seconds", "s"),
              ("model load time", "load_seconds", "s"),
              ("prompt tokens", "prompt_tokens", ""),
              ("cached prompt tokens", "cached_prompt_tokens", ""),
              ("prompt processing", "prompt_seconds", "s"),
              ("uncached prompt rate", "uncached_prompt_tokens_per_second", "tokens/s"),
              ("generated tokens", "completion_tokens", ""),
              ("generation duration", "decode_seconds", "s"),
              ("GENERATION RATE", "generation_tokens_per_second", "tokens/s"),
              ("client stream estimate", "stream_estimated_tokens_per_second", "tokens/s")]
    for title, key, unit in fields:
        value = result.get(key)
        if value is not None:
            print(f"  {title:25} {value:10.2f} {unit}")
    print(f"  finish reason: {result['finish_reason']}; rate source: {result['generation_rate_source']}")
    for warning in result["warnings"]:
        print(f"  Note: {warning}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("engine", choices=URLS)
    prompt = parser.add_mutually_exclusive_group()
    prompt.add_argument("--prompt", default=None)
    prompt.add_argument("--prompt-file", type=Path)
    parser.add_argument("--start-server", action="store_true", help="Start the selected local server if needed; stop it on exit.")
    parser.add_argument("--startup-timeout", type=float, default=1200,
                        help="Server readiness timeout in seconds (1200, matching the organizer notebook).")
    parser.add_argument("--runs", type=int, default=3, help="Measured requests after a separate short warm-up (default: 3).")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Output-token ceiling per benchmark request (default: 1024).")
    parser.add_argument("--context", type=int, default=32768, help="Ollama context; must match the running vLLM configuration (default: 32768).")
    parser.add_argument("--temperature", type=float, default=1.0, help="Default 1.0, matching chat_12b.sh's model settings.")
    parser.add_argument("--think", action="store_true", help="Enable thinking on both engines; default off, matching chat_12b.sh.")
    parser.add_argument("--show-output", action="store_true", help="Print generated answers while streaming.")
    parser.add_argument("--reference-tps", type=float, default=35.48, help="Historical Ollama generation rate for reference (default: 35.48).")
    args = parser.parse_args()
    if args.runs < 1 or not 2 <= args.max_tokens <= args.context:
        parser.error("Use --runs >= 1 and 2 <= --max-tokens <= context (the prompt must also fit).")
    if args.reference_tps <= 0 or args.temperature < 0:
        parser.error("Reference speed must be positive; temperature must be nonnegative.")
    if args.startup_timeout <= 0:
        parser.error("--startup-timeout must be positive.")
    text = args.prompt_file.read_text() if args.prompt_file else (args.prompt or DEFAULT_PROMPT)
    if not text.strip():
        parser.error("The prompt cannot be empty.")
    if args.engine == "ollama" and ready("vllm"):
        raise RuntimeError("Stop the vLLM server first so only one model occupies GPU memory during this benchmark.")
    loaded = ollama_loaded()
    if args.engine == "vllm" and loaded:
        raise RuntimeError("Unload Ollama's loaded models first with the local Ollama 'stop MODEL' command.")
    if args.engine == "ollama" and any(m.get("name", "").split(":")[0] != MODELS["ollama"] for m in loaded):
        raise RuntimeError("Unload other Ollama models before benchmarking gemma4-12b-chat.")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory = ROOT / "results" / f"{stamp}-speed-{args.engine}"
    directory.mkdir(parents=True)
    print(f"Results: {directory}", flush=True)
    server = server_log = None
    loaded_by_benchmark = False
    report = {"engine": args.engine, "model": MODELS[args.engine],
              "settings": {**vars(args), "prompt_file": str(args.prompt_file) if args.prompt_file else None},
              "prompt": text, "runs": [], "status": "starting",
              "comparison_note": "Installed setups use different quantized model artifacts: Ollama NVFP4 vs vLLM MLX affine 4-bit with mixed precision. This is not an engine-only comparison."}
    try:
        if not ready(args.engine):
            if not args.start_server:
                raise RuntimeError(f"Server is stopped. Add --start-server or run scripts/start_{args.engine}.sh in another terminal.")
            server_log = (directory / "server.log").open("w")
            server = subprocess.Popen(["sh", str(ROOT / "scripts" / f"start_{args.engine}.sh")],
                                      cwd=ROOT, stdout=server_log, stderr=subprocess.STDOUT, start_new_session=True)
            print("Starting local server; startup is excluded from measured runs...", flush=True)
            deadline = time.monotonic() + args.startup_timeout
            while not ready(args.engine):
                if server.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError(f"Server did not become ready. See {directory / 'server.log'}")
                time.sleep(1)
        if args.engine == "ollama":
            info = json_request(URLS["ollama"] + "/api/show", {"model": MODELS["ollama"]})
            report["model_metadata"] = {key: info.get(key) for key in ("details", "parameters", "model_info")}
            report["server_version"] = json_request(URLS["ollama"] + "/api/version")
            loaded_by_benchmark = not loaded
        else:
            report["model_metadata"] = json_request(URLS["vllm"] + "/v1/models")
            report["local_runtime_configuration"] = json.loads((ROOT / "dev/vllm-runtime.json").read_text())
            model = next((m for m in report["model_metadata"].get("data", []) if m["id"] == MODELS["vllm"]), None)
            if model is None:
                raise RuntimeError("The expected gemma4-12b-local model is not served.")
            if model.get("max_model_len") is not None and model["max_model_len"] != args.context:
                raise RuntimeError(f"vLLM context is {model['max_model_len']}; use the same --context for both benchmarks.")
        print(f"Thinking: {args.think}; context: {args.context}; max output: {args.max_tokens}; temperature: {args.temperature}")
        print(report["comparison_note"])
        print("Warming the model with a separate short prompt...", flush=True)
        report["warmup"] = generate(args.engine, [{"role": "user", "content": "Reply with the word Ready."}], args, directory, "warmup", 16)
        report["status"] = "running"
        for n in range(1, args.runs + 1):
            print(f"\nRun {n}/{args.runs} (fresh conversation; server prefix cache may be reused)...", flush=True)
            result = generate(args.engine, [{"role": "user", "content": text}], args, directory, f"run-{n}", args.max_tokens)
            report["runs"].append(result)
            show(result)
        speeds = [r["generation_tokens_per_second"] for r in report["runs"] if r["generation_tokens_per_second"] is not None]
        if len(speeds) != args.runs:
            raise RuntimeError("Not all runs supplied a usable generation rate.")
        median = statistics.median(speeds)
        report["median_generation_tokens_per_second"] = median
        report["historical_reference_ratio"] = median / args.reference_tps
        report["status"] = "completed"
        print(f"\nMedian generation: {median:.2f} tokens/s")
        print(f"Range: {min(speeds):.2f}–{max(speeds):.2f} tokens/s")
        print(f"Relative to your historical Ollama {args.reference_tps:.2f} tokens/s: {median / args.reference_tps:.2f}x the generation rate.")
        print("For a controlled comparison, run both engines with the same arguments. Original chat history is not reproduced.")
    except BaseException as exc:
        report["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "error"
        report["error"] = str(exc) or type(exc).__name__
        raise
    finally:
        (directory / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        if loaded_by_benchmark:
            try:
                json_request(URLS["ollama"] + "/api/generate", {"model": MODELS["ollama"], "keep_alive": 0})
            except (OSError, RuntimeError):
                print("Could not unload the benchmark model; use Ollama stop if needed.", file=sys.stderr)
        if server is not None and server.poll() is None:
            os.killpg(server.pid, signal.SIGTERM)
            try:
                server.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait()
        if server_log:
            server_log.close()
        print(f"Saved benchmark artifacts: {directory}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nBenchmark interrupted.", file=sys.stderr)
        sys.exit(130)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
