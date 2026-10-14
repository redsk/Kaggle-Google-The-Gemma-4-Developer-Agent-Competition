"""Scripted HTTP responses through real ADK/LiteLLM; no inference or sandbox.

Only the HTTP transport and external verifier are replaced. Tool execution,
tool-result follow-up, summarization, streaming and client retries are real.
"""
import asyncio
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from litellm.llms.custom_httpx.aiohttp_transport import LiteLLMAiohttpTransport
from google.adk.agents import Agent
from google.adk.apps.llm_event_summarizer import LlmEventSummarizer
from google.adk.events import Event
from google.adk.models.lite_llm import LiteLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.runners import InMemoryRunner
from google.adk.tools import FunctionTool
from google.genai import types

from observability import Telemetry, instrument
from vllm_client import VllmRemainingContextClient

FIXTURE_SECRET = "synthetic-secret-never-export-this"
ENDPOINT = "http://fixture.invalid/v1"


def content(text):
    return types.Content(role="user", parts=[types.Part(text=text)])


async def exercise(directory, *, endpoint=None):
    directory = Path(directory)
    telemetry = Telemetry(directory, endpoint=endpoint, project="gemma-observability-preflight",
                          endpoints=[ENDPOINT], secrets=[FIXTURE_SECRET], metadata={
                              "scope": "scripted transport; no real model or competition data",
                              "backend": "synthetic-vllm-compatible", "model": "gemma4-12b-local"})
    requests = []
    retry_count = 0
    tool_reply_sent = False

    def response(payload, *, message=None, status=200):
        return httpx.Response(status, json=payload or {
            "id": "fixture-response", "object": "chat.completion", "created": 1,
            "model": "gemma4-12b-local", "choices": [{"index": 0,
                "finish_reason": "tool_calls" if message and message.get("tool_calls") else "stop",
                "message": message or {"role": "assistant", "content": "FIXTURE_OK"}}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
        })

    async def transport(self, request):
        nonlocal retry_count, tool_reply_sent
        body = json.loads(request.content)
        requests.append(body)
        assert len(requests) <= 12, "Scripted fixture exceeded its expected request sequence"
        assert str(request.url).startswith(ENDPOINT), "Unexpected network request in offline fixture"
        messages = body["messages"]
        text = str(messages)
        if "RETRY_FIXTURE" in text:
            retry_count += 1
            if retry_count == 1:
                return response({"error": {"message": "synthetic temporary failure", "type": "server_error"}}, status=503)
        if body.get("stream"):
            chunks = [
                {"id": "fixture-stream", "object": "chat.completion.chunk", "created": 1, "model": "gemma4-12b-local",
                 "choices": [{"index": 0, "delta": {"role": "assistant", "content": "STREAM_OK"}, "finish_reason": None}]},
                {"id": "fixture-stream", "object": "chat.completion.chunk", "created": 1, "model": "gemma4-12b-local",
                 "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                 "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}},
            ]
            data = "".join("data: " + json.dumps(c) + "\n\n" for c in chunks) + "data: [DONE]\n\n"
            return httpx.Response(200, content=data, headers={"content-type": "text/event-stream"})
        if body.get("tools") and not tool_reply_sent:
            tool_reply_sent = True
            calls = [("write_file", {"text": "changed by a real fixture tool"}),
                     ("get_code_subgraph", {"node": "fixture"}), ("fail_tool", {"value": "missing"})]
            return response(None, message={"role": "assistant", "content": None, "tool_calls": [
                {"id": f"call-{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
                for i, (name, args) in enumerate(calls)]})
        return response(None)

    def write_file(text: str) -> dict:
        """Write a synthetic fixture file."""
        (directory / "fixture.txt").write_text(text)
        return {"status": "success"}

    def get_code_subgraph(node: str) -> dict:
        """Return a synthetic code graph."""
        return {"nodes": [node], "edges": []}

    def fail_tool(value: str) -> dict:
        """Return a synthetic tool error."""
        return {"status": "error", "error": "FIXTURE_NOT_FOUND", "value": value}

    async def fake_verify(*args, **kwargs):
        return SimpleNamespace(resolved=True, test_exit_code=0, error=None, test_output="synthetic verification: 1 passed")

    model = LiteLlm(model="openai/gemma4-12b-local", api_base=ENDPOINT, api_key=FIXTURE_SECRET,
                   num_retries=1, llm_client=VllmRemainingContextClient(32768))
    eval_module = importlib.import_module("swegemma.evaluate")
    with patch.object(httpx.AsyncHTTPTransport, "handle_async_request", transport), \
         patch.object(LiteLLMAiohttpTransport, "handle_async_request", transport), \
         patch.object(eval_module, "verify_task", fake_verify), telemetry.run("synthetic.fixture"), instrument():
        agent = Agent(name="fixture_agent", model=model,
                      tools=[write_file, get_code_subgraph, fail_tool],
                      generate_content_config=types.GenerateContentConfig(max_output_tokens=32768))
        runner = InMemoryRunner(agent=agent, app_name="observability_fixture")
        session = await runner.session_service.create_session(app_name="observability_fixture", user_id="fixture")
        events = [event async for event in runner.run_async(
            user_id="fixture", session_id=session.id, new_message=content("Execute the three fixture tools."))]
        assert any(e.is_final_response() for e in events)
        summary = await LlmEventSummarizer(model).maybe_summarize_events(events=events)
        assert summary.actions.compaction
        streamed = [r async for r in model.generate_content_async(LlmRequest(contents=[content("STREAM_FIXTURE")]), stream=True)]
        assert any(r.content and any(p.text == "STREAM_OK" for p in r.content.parts) for r in streamed)
        retried = [r async for r in model.generate_content_async(LlmRequest(contents=[content("RETRY_FIXTURE")]))]
        assert retried and retry_count == 2
        verified = await eval_module.verify_task(agent_patch="synthetic patch; never applied to a repository")
        assert verified.resolved
        # Exceptions must propagate unchanged and close their spans.
        def explode(value: str) -> dict:
            raise ValueError(FIXTURE_SECRET)
        try:
            await FunctionTool(explode).run_async(args={"value": "fixture"}, tool_context=SimpleNamespace(function_call_id="exception"))
        except ValueError as error:
            assert str(error) == FIXTURE_SECRET
        else:
            raise AssertionError("Instrumentation swallowed a tool exception")

    assert (directory / "fixture.txt").read_text() == "changed by a real fixture tool"
    assert all("max_completion_tokens" not in r and "max_tokens" not in r for r in requests)
    logs = [json.loads(line) for line in (directory / "telemetry.jsonl").read_text().splitlines()]
    spans = [json.loads(line) for line in (directory / "spans.jsonl").read_text().splitlines()]
    assert not any(FIXTURE_SECRET in p.read_text() for p in directory.glob("*.json*"))
    names = [s["name"] for s in spans]
    assert names.count("llm.compaction") == 1
    assert names.count("history.compaction") == 1
    assert names.count("llm.agent") == 4
    assert names.count("model.http") == 6
    assert names.count("tool.get_code_subgraph") == 1
    assert names.count("patch.verification") == 1
    assert next(s for s in spans if s["name"] == "tool.fail_tool")["status"]["status_code"] == "ERROR"
    verify_span = next(s for s in spans if s["name"] == "patch.verification")
    root_span = next(s for s in spans if s["name"] == "synthetic.fixture")
    assert verify_span["context"]["trace_id"] != root_span["context"]["trace_id"]
    assert verify_span["links"][0]["context"]["trace_id"] == root_span["context"]["trace_id"]
    starts = {e["span_id"] for e in logs if e["event"] == "span.start"}
    ends = {e["span_id"] for e in logs if e["event"] == "span.end"}
    assert starts == ends, "Unclosed spans"
    return {"status": "passed", "scope": "scripted HTTP, real ADK/LiteLLM; no inference", "spans": len(spans),
            "http_attempts": len(requests), "counts": telemetry.counts,
            "trace_id": root_span["context"]["trace_id"], "directory": str(directory)}
