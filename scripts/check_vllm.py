"""Check local vLLM output limits, thinking budget, and tool-call parsing."""

import json
from pathlib import Path
import time

import requests
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
BASE = "http://127.0.0.1:8000"
MODEL = "gemma4-12b-local"


def call(label, **kwargs):
    start = time.perf_counter()
    response = requests.post(BASE + "/v1/chat/completions", json={
        "model": MODEL, "temperature": 0, "stream": False, **kwargs,
    }, timeout=(10, 300))
    response.raise_for_status()
    data = response.json()
    (ROOT / "results" / ("vllm-preflight-" + label + ".json")).write_text(json.dumps(data, indent=2) + "\n")
    return data, round(time.perf_counter() - start, 2)


def main():
    requests.get(BASE + "/health", timeout=10).raise_for_status()
    tokenizer = Tokenizer.from_file(str(ROOT / "models/vllm/gemma-4-12B-it-qat-4bit/tokenizer.json"))
    result = {}
    data, seconds = call("output-limit", max_completion_tokens=32,
        chat_template_kwargs={"enable_thinking": False},
        messages=[{"role": "user", "content": "List the integers from 1 to 1000, separated by spaces."}])
    result["output_limit"] = {"seconds": seconds, "completion_tokens": data["usage"]["completion_tokens"],
                              "finish_reason": data["choices"][0]["finish_reason"]}
    assert data["usage"]["completion_tokens"] == 32, result
    print(json.dumps({"output_limit": result["output_limit"]}), flush=True)
    data, seconds = call("thinking-budget", max_completion_tokens=256, thinking_token_budget=16,
        chat_template_kwargs={"enable_thinking": True},
        messages=[{"role": "user", "content": "Calculate 1739 times 2843. Think carefully and give only the final integer after reasoning."}])
    message = data["choices"][0]["message"]
    reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
    count = len(tokenizer.encode(reasoning, add_special_tokens=False).ids)
    result["thinking_budget"] = {"seconds": seconds, "requested": 16, "reasoning_tokens": count,
                                 "within_budget_with_marker_allowance": count <= 20,
                                 "answer_present": bool(message.get("content")),
                                 "completion_tokens": data["usage"]["completion_tokens"]}
    print(json.dumps({"thinking_budget": result["thinking_budget"]}), flush=True)
    data, seconds = call("tool-call", max_completion_tokens=512, thinking_token_budget=32,
        chat_template_kwargs={"enable_thinking": True},
        messages=[{"role": "user", "content": "Call get_status now. Do not explain."}],
        tools=[{"type": "function", "function": {"name": "get_status", "description": "Return task status.",
                "parameters": {"type": "object", "properties": {}, "required": []}}}])
    calls = data["choices"][0]["message"].get("tool_calls", [])
    result["tool_call"] = {"seconds": seconds, "names": [c["function"]["name"] for c in calls],
                           "completion_tokens": data["usage"]["completion_tokens"]}
    assert "get_status" in result["tool_call"]["names"], result
    for entry in calls:
        json.loads(entry["function"]["arguments"])
    print(json.dumps({"tool_call": result["tool_call"]}), flush=True)
    (ROOT / "results/vllm-preflight.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
