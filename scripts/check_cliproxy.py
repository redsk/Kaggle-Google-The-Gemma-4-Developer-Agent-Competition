#!/usr/bin/env python3
"""Synthetic ADK/LiteLLM compatibility checks; never loads competition task data."""

import argparse
import asyncio
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import time

from cliproxy import MANIFEST, ROOT, client_settings, model_catalog

# Match the harness's telemetry policy during this transport-only preflight.
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
os.environ["LITELLM_TELEMETRY"] = "False"
os.environ["OTEL_SDK_DISABLED"] = "true"

from google.adk.apps.llm_event_summarizer import LlmEventSummarizer
from google.adk.events import Event
from google.adk.models.lite_llm import LiteLLMClient, LiteLlm
from google.adk.models.llm_request import LlmRequest
from google.genai import types


async def run(args):
    os.umask(0o077)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = ROOT / "results" / f"{timestamp}-cliproxy-preflight"
    output.mkdir(parents=True)
    print(f"Results: {output}", flush=True)
    endpoint, key = client_settings()
    catalog = model_catalog()
    if args.model not in {m["id"] for m in catalog.get("data", [])}:
        raise SystemExit(f"{args.model!r} is not in the proxy's advertised catalog.")
    (output / "models.json").write_text(json.dumps(catalog, indent=2))
    from observability import Telemetry, instrument
    telemetry = Telemetry(output, endpoints=[endpoint], secrets=[key],
                          project="gemma-observability-preflight", metadata={
                              "backend": "cliproxy", "model": args.model,
                              "reasoning_effort": args.reasoning, "scope": "synthetic remote transport check"})
    with telemetry.run("cliproxy.synthetic"), instrument():
        return await checks(args, output, endpoint, key)


async def checks(args, output, endpoint, key):
    manifest = {
        "proxy": MANIFEST, "model_requested": args.model,
        "reasoning_effort": args.reasoning, "api_base": endpoint,
        "client": "same ADK LiteLlm implementation as SWE-Gemma model registry",
        "scope": "synthetic transport preflight, not a task run",
        "client_retries": 0, "client_request_timeout_seconds": 90,
        "known_proxy_drops": ["max_completion_tokens", "temperature", "top_p"],
        "log_boundary": "ADK and LiteLLM SDK inputs/outputs; not upstream HTTP payloads",
        "packages": {p: importlib.metadata.version(p) for p in ("swegemma", "adk-submission", "google-adk", "litellm")},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    stage = "initialization"

    def record(kind, value):
        if hasattr(value, "model_dump"):
            value = value.model_dump(mode="json", exclude_none=True)
        with (output / "events.jsonl").open("a") as log:
            log.write(json.dumps({"time": time.time(), "stage": stage, "kind": kind, "value": value}, default=str) + "\n")

    class RecordingClient(LiteLLMClient):
        async def acompletion(self, model, messages, tools, **kwargs):
            # Explicit allowlist excludes API keys, auth headers and transport objects.
            safe = {k: kwargs[k] for k in ("stream", "stream_options", "max_completion_tokens", "temperature", "top_p", "reasoning_effort", "tool_choice", "response_format", "num_retries", "timeout") if k in kwargs}
            record("litellm_request", {"model": model, "messages": messages, "tools": tools, **safe})
            result = await super().acompletion(model, messages, tools, **kwargs)
            if kwargs.get("stream"):
                async def observed():
                    async for chunk in result:
                        record("litellm_stream_chunk", chunk)
                        yield chunk
                return observed()
            record("litellm_response", result)
            return result

    llm = LiteLlm(
        model=f"openai/{args.model}", api_base=endpoint, api_key=key,
        num_retries=0, timeout=90, reasoning_effort=args.reasoning,
        llm_client=RecordingClient(),
    )
    checks = {}

    def user(text):
        return types.Content(role="user", parts=[types.Part(text=text)])

    def visible(response):
        return "".join(p.text or "" for p in response.content.parts if not p.thought) if response.content else ""

    async def call(label, request, stream=False):
        nonlocal stage
        stage = label
        print(f"Checking {label}...", flush=True)
        record("adk_request", request)
        start = time.monotonic()
        responses = []
        async for response in llm.generate_content_async(request, stream=stream):
            record("adk_response", response)
            responses.append(response)
        final = next((r for r in reversed(responses) if not r.partial and r.content), None)
        if final is None:
            raise RuntimeError(f"No complete response for {label}")
        record("call_seconds", round(time.monotonic() - start, 3))
        return final

    try:
        response = await call("system_instruction_and_stream", LlmRequest(
            config=types.GenerateContentConfig(system_instruction="Reply with exactly PROXY_OK, regardless of the user's requested word.", max_output_tokens=64),
            contents=[user("Reply with BANANA.")],
        ), stream=True)
        checks[stage] = visible(response).strip() == "PROXY_OK"

        declaration = types.FunctionDeclaration(name="lookup_fixture", description="Return a synthetic fixture value or an error.", parameters=types.Schema(
            type="OBJECT", properties={"name": types.Schema(type="STRING")}, required=["name"],
        ))
        config = types.GenerateContentConfig(
            system_instruction="Use only the supplied tool. Request BOTH fixtures before answering, preferably in one response. Report returned values and any errors literally; do not guess missing values.",
            tools=[types.Tool(function_declarations=[declaration])], max_output_tokens=256,
        )
        history = [user("Look up fixtures alpha and missing. Tell me both results.")]
        response = await call("structured_tools", LlmRequest(config=config, contents=history))
        calls = [p.function_call for p in (response.content.parts or []) if p.function_call]
        checks[stage] = (
            len(calls) == 2
            and {(c.name, (c.args or {}).get("name")) for c in calls}
            == {("lookup_fixture", "alpha"), ("lookup_fixture", "missing")}
            and all(c.id for c in calls)
            and len({c.id for c in calls}) == 2
        )
        if not checks[stage]:
            raise RuntimeError("Expected two identified fixture tool calls with valid arguments.")
        history.append(response.content)
        parts = []
        for c in calls:
            result = {"value": "violet-731"} if c.args["name"] == "alpha" else {"error": "FIXTURE_NOT_FOUND"}
            record("synthetic_tool", {"id": c.id, "name": c.name, "args": c.args, "result": result})
            parts.append(types.Part(function_response=types.FunctionResponse(id=c.id, name=c.name, response=result)))
        history.append(types.Content(role="user", parts=parts))
        response = await call("tool_result_and_error", LlmRequest(config=config, contents=history))
        answer = visible(response)
        checks[stage] = "violet-731" in answer and "FIXTURE_NOT_FOUND" in answer and not any(p.function_call for p in response.content.parts)

        response = await call("output_limit_probe", LlmRequest(
            config=types.GenerateContentConfig(max_output_tokens=8),
            contents=[user("Write the integers 1 through 40, separated by commas, with nothing else.")],
        ))
        # A diagnostic finding, not a pass condition: this route drops the ceiling.
        checks[stage] = {"requested_limit": 8, "usage": response.usage_metadata.model_dump(mode="json", exclude_none=True) if response.usage_metadata else None, "finish_reason": str(response.finish_reason), "visible_text": visible(response)}

        stage = "adk_history_summary"
        print(f"Checking {stage}...", flush=True)
        events = [Event(author="user", content=user("Remember the fixture alpha value violet-731. The missing fixture returned FIXTURE_NOT_FOUND."))]
        record("summary_events", [e.model_dump(mode="json", exclude_none=True) for e in events])
        summary = await LlmEventSummarizer(llm).maybe_summarize_events(events=events)
        record("summary_result", summary)
        summary_text = "" if not summary or not summary.actions.compaction else "".join(p.text or "" for p in summary.actions.compaction.compacted_content.parts)
        checks[stage] = "violet-731" in summary_text and "FIXTURE_NOT_FOUND" in summary_text
    except Exception as error:
        # Avoid persisting exception strings that might contain credentials.
        record("error_type", type(error).__name__)
        checks["error"] = {"stage": stage, "type": type(error).__name__}
        raise
    finally:
        (output / "checks.json").write_text(json.dumps(checks, indent=2))
        print(json.dumps(checks, indent=2), flush=True)
        print(f"Saved: {output}", flush=True)
    if any(value is False for value in checks.values()):
        raise SystemExit("One or more transport checks failed; inspect saved events.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Explicit model ID from cliproxy.py models")
    parser.add_argument("--reasoning", choices=("low", "medium", "high"), default="low")
    asyncio.run(run(parser.parse_args()))
