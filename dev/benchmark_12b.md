# Local 12B speed comparison

Run from the repository root:

```sh
.venv/bin/python scripts/benchmark_12b.py vllm --start-server
.venv/bin/python scripts/benchmark_12b.py ollama --start-server
```

Run the commands sequentially. Both use the same default neural-network question.
`--start-server` starts the selected local server only if necessary and stops a
server it started when the benchmark finishes, including after Ctrl+C. Existing
servers are left running. If vLLM was already running, stop it before benchmarking
Ollama. The script checks that the other engine has no loaded model. An Ollama
model loaded by this benchmark is unloaded afterward; an already-loaded one is
left loaded. No Kaggle job or competition task is involved.

Use your own question with identical arguments for both commands:

```sh
.venv/bin/python scripts/benchmark_12b.py vllm --start-server --prompt "Explain backpropagation." --show-output
.venv/bin/python scripts/benchmark_12b.py ollama --start-server --prompt "Explain backpropagation." --show-output
```

For a longer question, use `--prompt-file path/to/question.txt`. Each request is a
fresh conversation; prior interactive chat history is not included.

## Defaults and measurements

- One separate short warm-up, followed by three measured requests (`--runs 3`).
- Thinking off, matching `scripts/chat_12b.sh`; `--think` enables it explicitly.
- 32,768 context tokens on both engines. This matches the current vLLM server;
  the casual Ollama chat alias normally uses 8,192. The override applies only to
  these benchmark requests. The vLLM server's context is checked when exposed.
- Temperature 1, top-p 0.95, top-k 64, seed 42 on both engines. Quantization and
  backend differences mean identical seeds do not guarantee identical answers.
- Maximum 1,024 generated tokens per request (`--max-tokens`, up to 4,096).
  A `length` finish reason means the benchmark output ceiling was reached.
- Repeated requests can reuse prefix caches. Inspect the first run separately
  when comparing first-token latency with later runs. No cache is forcibly reset.
- Full response events, request settings, answers, per-run metrics, server startup
  logs when applicable, and `summary.json` are saved in `results/*-speed-ENGINE/`.

`GENERATION RATE` is the rate to compare with Ollama's reported `eval rate`:

- Ollama: generated tokens divided by `eval_duration`, converted from nanoseconds.
- vLLM: generated tokens minus the first token, divided by the measured decode
  duration from `/metrics`. The first generated token belongs to prefill.
- If isolated vLLM server metrics cannot be verified, the script explicitly uses
  a client streaming estimate. It does not count network chunks as tokens.

Prompt throughput excludes cached tokens when the server exposes that count.
The script also reports elapsed wall time, time to first streamed output, and a
client streaming estimate. With thinking enabled, first output may be thinking;
time to first answer is separately recorded in the JSON results.

The displayed ratio against 35.48 tokens/s refers to the historical Ollama result
provided in chat. It is a reference, not a replay of its 901-token conversation.
Compare the two fresh runs for the controlled prompt comparison.

## Interpretation

This compares the installed local setups. Ollama 0.40.4 reports an MLX runner
with NVFP4 weights; the vLLM-Metal checkpoint is a different MLX affine
quantization with mixed precision. Any measured difference includes model
representation and runtime settings, not just the serving engine. These results
also do not measure performance on a long coding-agent conversation.

Thermal throttling was reproduced on this Mac: GPU clocks and output speed fell
together during repeated requests, then speed recovered after cooling. See the
[investigation and measurements](slowdown-investigation-2026-10-14.md). Compare
results under similar cooling conditions; a median from a slowing sequence is
not a stable engine throughput estimate.

To collect hardware diagnostics alongside a benchmark:

```sh
.venv/bin/python scripts/monitor_local_inference.py \
  --label vllm --macmon downloads/tools/macmon-v0.9.0 -- \
  .venv/bin/python scripts/benchmark_12b.py vllm --start-server
```

Omit `--macmon PATH` to collect only the built-in macOS counters (no GPU clock
or temperature readings). The wrapper does not change model or fan settings.

Metric definitions: [Ollama usage](https://docs.ollama.com/api/usage),
[vLLM metrics](https://docs.vllm.ai/en/latest/usage/metrics/).
