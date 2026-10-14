"""Local Phoenix/OTLP traces for the pinned harness; never changes model inputs.

Uses the OpenTelemetry packages already supplied by the official wheelhouse and
OpenInference attributes. Scoped wrappers are restored on exit. No SDK upgrade,
provider-side tools, automatic eval calls, sampling, or new execution budget.
"""

from contextlib import contextmanager, ExitStack
from contextvars import ContextVar
from functools import wraps
import importlib
import json
import os
from pathlib import Path
import threading
import time
from urllib.parse import urlparse
from unittest.mock import patch

from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider, SpanLimits
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor, SpanExporter, SpanExportResult
from opentelemetry.sdk.trace.sampling import ALWAYS_ON
from opentelemetry.trace import Link, Status, StatusCode

DEFAULT_ENDPOINT = "http://127.0.0.1:6006/v1/traces"
DEFAULT_PROJECT = "gemma-agent-comparison"
PHASE = ContextVar("gemma_trace_phase", default="agent")
ACTIVE = ContextVar("gemma_telemetry", default=None)
SECRET_FIELDS = {"api_key", "authorization", "proxy_authorization", "cookie", "set_cookie",
                 "access_token", "refresh_token", "id_token", "client_secret", "password"}
GRAPH_TOOLS = {"get_code_neighbors", "search_similar_code", "get_code_subgraph"}


def plain(value):
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump(mode="json", exclude_none=True)
        except Exception as error:
            return {"serialization_error": type(error).__name__, "object_type": type(value).__name__}
    if isinstance(value, Path):
        return str(value)
    return value


class JsonSpanExporter(SpanExporter):
    def __init__(self, telemetry):
        self.telemetry = telemetry

    def export(self, spans):
        for span in spans:
            self.telemetry.write("spans.jsonl", json.loads(span.to_json()))
        return SpanExportResult.SUCCESS

    def shutdown(self):
        pass


class Telemetry:
    def __init__(self, directory, *, metadata=None, endpoints=(), secrets=(),
                 endpoint=DEFAULT_ENDPOINT, project=DEFAULT_PROJECT):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.metadata = {"cost_semantics": "UI dollar amounts may be API-price estimates; not local or subscription billing",
                         **(metadata or {})}
        self.secrets = {s for s in secrets if isinstance(s, str) and len(s) > 5}
        self.origins = {(urlparse(u).scheme, urlparse(u).netloc) for u in endpoints}
        self.lock = threading.Lock()
        self.sequence = 0
        self.counts = {"model_calls": 0, "summary_calls": 0, "tools": 0, "graph_tools": 0, "http_attempts": 0}
        self.root = None
        # Enable only this explicit local provider, regardless of inherited env.
        previous = os.environ.get("OTEL_SDK_DISABLED")
        os.environ["OTEL_SDK_DISABLED"] = "false"
        try:
            self.provider = TracerProvider(sampler=ALWAYS_ON, span_limits=SpanLimits(
                max_span_attributes=SpanLimits.UNSET, max_span_attribute_length=SpanLimits.UNSET,
            ), resource=Resource({
                "service.name": "gemma-competition-harness", "openinference.project.name": project,
            }))
        finally:
            if previous is None:
                os.environ.pop("OTEL_SDK_DISABLED", None)
            else:
                os.environ["OTEL_SDK_DISABLED"] = previous
        self.provider.add_span_processor(SimpleSpanProcessor(JsonSpanExporter(self)))
        if endpoint:
            parsed = urlparse(endpoint)
            if parsed.hostname not in {"127.0.0.1", "localhost", "::1"} or parsed.scheme != "http":
                raise ValueError("This setup exports traces only to a local Phoenix endpoint")
            import requests
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            session = requests.Session()
            session.trust_env = False
            exporter = OTLPSpanExporter(endpoint=endpoint, timeout=5, session=session,
                                       headers={"x-gemma-tracing": "local"})
            self.provider.add_span_processor(BatchSpanProcessor(exporter))
        self.tracer = self.provider.get_tracer("gemma.observability", "1")
        self.write("telemetry.json", {"project": project, "endpoint": endpoint, "run_id": self.directory.name,
                   "metadata": self.metadata, "capture_boundary": "ADK, LiteLLM SDK and HTTP to configured endpoints; not proxy-to-provider traffic",
                   "sampling": "all spans", "raw_log": "telemetry.jsonl", "span_log": "spans.jsonl"})

    def clean(self, value):
        value = plain(value)
        if isinstance(value, dict):
            return {str(k): "[REDACTED]" if str(k).lower().replace("-", "_") in SECRET_FIELDS else self.clean(v)
                    for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self.clean(v) for v in value]
        if isinstance(value, str):
            for secret in self.secrets:
                value = value.replace(secret, "[REDACTED]")
            return value
        return value

    def dumps(self, value):
        return json.dumps(self.clean(value), default=str, ensure_ascii=False)

    def write(self, filename, value):
        # Export failures must never change the agent's answer or tool behavior.
        try:
            line = self.dumps(value)
            with self.lock, (self.directory / filename).open("a", encoding="utf-8") as output:
                output.write(line + "\n")
        except Exception as error:
            import sys
            print(f"Trace write failed: {type(error).__name__}", file=sys.__stderr__)

    def record(self, event, *, span=None, **fields):
        ctx = (span or trace.get_current_span()).get_span_context()
        with self.lock:
            self.sequence += 1
            number = self.sequence
        self.write("telemetry.jsonl", {"time_ns": time.time_ns(), "monotonic_ns": time.monotonic_ns(),
                   "sequence": number, "run_id": self.directory.name, "event": event,
                   "trace_id": f"{ctx.trace_id:032x}", "span_id": f"{ctx.span_id:016x}", **fields})

    def start(self, name, kind="CHAIN", *, input=None, attributes=None, separate=False):
        attrs = {"openinference.span.kind": kind, "session.id": self.directory.name,
                 "metadata": self.dumps(self.metadata), "gemma.run_id": self.directory.name,
                 "gemma.phase": PHASE.get(), **(attributes or {})}
        if input is not None:
            attrs.update({"input.value": self.dumps(input), "input.mime_type": "application/json"})
        links = [Link(self.root.get_span_context())] if separate and self.root else []
        span = self.tracer.start_span(name, attributes=attrs, context=Context() if separate else None, links=links)
        self.record("span.start", span=span, name=name, input=input)
        return span

    def finish(self, span, *, output=None, error=None):
        if not span.is_recording():
            return
        if output is not None:
            span.set_attribute("output.value", self.dumps(output))
            span.set_attribute("output.mime_type", "application/json")
        if error is not None:
            # Exception strings may contain authorization headers or full URLs.
            span.set_status(Status(StatusCode.ERROR, type(error).__name__))
            span.add_event("exception", {"exception.type": type(error).__name__})
        self.record("span.end", span=span, name=span.name, output=output,
                    error_type=type(error).__name__ if error else None)
        span.end()

    @contextmanager
    def span(self, name, kind="CHAIN", **kwargs):
        span = self.start(name, kind, **kwargs)
        with trace.use_span(span, end_on_exit=False, record_exception=False, set_status_on_exception=False):
            try:
                yield span
            except BaseException as error:
                self.finish(span, error=error)
                raise
            finally:
                self.finish(span)

    @contextmanager
    def run(self, name="task.run"):
        token = ACTIVE.set(self)
        try:
            with self.span(name, "AGENT", input=self.metadata) as span:
                self.root = span
                self.write("trace-link.json", {"run_id": self.directory.name,
                           "trace_id": f"{span.get_span_context().trace_id:032x}"})
                try:
                    yield self
                finally:
                    span.set_attribute("gemma.counts", self.dumps(self.counts))
        finally:
            ACTIVE.reset(token)
            self.provider.force_flush(timeout_millis=5000)
            self.provider.shutdown()

    def model_output(self, span, response):
        usage = getattr(response, "usage_metadata", None)
        if usage:
            for source, target in (("prompt_token_count", "prompt"), ("candidates_token_count", "completion"),
                                   ("total_token_count", "total")):
                value = getattr(usage, source, None)
                if value is not None:
                    span.set_attribute("llm.token_count." + target, value)
            for source in ("thoughts_token_count", "cached_content_token_count"):
                value = getattr(usage, source, None)
                if value is not None:
                    span.set_attribute("gemma.usage." + source, value)
            span.set_attribute("gemma.usage.source", "ADK provider response; not billing data")
        if getattr(response, "finish_reason", None):
            span.set_attribute("gemma.finish_reason", str(response.finish_reason))
        if getattr(response, "model_version", None):
            span.set_attribute("gemma.response_model", response.model_version)


@contextmanager
def instrument():
    """Instrument real pinned entry points, including direct summarizer calls.

    Wrappers forward the same objects and exceptions. ContextVar scopes capture
    to a run, and ExitStack restores every method even after cancellation.
    """
    import httpx
    from google.adk.models.lite_llm import LiteLlm, LiteLLMClient
    from google.adk.apps.llm_event_summarizer import LlmEventSummarizer
    from google.adk.tools.function_tool import FunctionTool
    evaluate_module = importlib.import_module("swegemma.evaluate")
    original_model = LiteLlm.generate_content_async
    original_client = LiteLLMClient.acompletion
    original_tool = FunctionTool.run_async
    original_summary = LlmEventSummarizer.maybe_summarize_events
    original_verify = evaluate_module.verify_task
    original_send = httpx.AsyncClient.send

    @wraps(original_model)
    async def model(self, llm_request, stream=False):
        obs = ACTIVE.get()
        if obs is None:
            async for response in original_model(self, llm_request, stream=stream):
                yield response
            return
        obs.counts["model_calls"] += 1
        span = obs.start("llm." + PHASE.get(), "LLM", input=llm_request,
                         attributes={"llm.model_name": llm_request.model or self.model,
                                     "gemma.streaming": stream})
        started = time.perf_counter()
        responses = []
        first_output = False
        generator = original_model(self, llm_request, stream=stream)
        try:
            while True:
                # Do not hold OpenTelemetry context across a yield: the caller
                # may break after one response (the official summarizer does).
                with trace.use_span(span, end_on_exit=False, record_exception=False, set_status_on_exception=False):
                    try:
                        response = await anext(generator)
                    except StopAsyncIteration:
                        break
                responses.append(plain(response))
                obs.record("model.response", span=span, response=response)
                obs.model_output(span, response)
                if not first_output and response.content and response.content.parts:
                    first_output = True
                    span.set_attribute("gemma.first_response_seconds", time.perf_counter() - started)
                    if stream:
                        span.set_attribute("gemma.time_to_first_output_seconds", time.perf_counter() - started)
                if not response.partial:
                    obs.finish(span, output=responses)
                yield response
        except BaseException as error:
            obs.finish(span, output=responses, error=error)
            raise
        finally:
            await generator.aclose()
            obs.finish(span, output=responses)

    @wraps(original_client)
    async def client(self, model, messages, tools, **kwargs):
        obs = ACTIVE.get()
        if obs:
            key = kwargs.get("api_key")
            if isinstance(key, str) and len(key) > 5:
                obs.secrets.add(key)
            fields = ("stream", "stream_options", "max_completion_tokens", "max_tokens", "temperature",
                      "top_p", "top_k", "seed", "reasoning_effort", "extra_body", "tool_choice",
                      "response_format", "num_retries", "timeout", "api_base")
            request = {"model": model, "messages": messages, "tools": tools,
                       **{k: kwargs[k] for k in fields if k in kwargs}}
            obs.record("client.request", request=request)
            span = trace.get_current_span()
            span.set_attribute("llm.invocation_parameters", obs.dumps({k: request[k] for k in fields if k in request}))
            for i, message in enumerate(messages):
                prefix = f"llm.input_messages.{i}.message."
                span.set_attribute(prefix + "role", message.get("role", "unknown"))
                if message.get("content") is not None:
                    content = message["content"]
                    span.set_attribute(prefix + "content", obs.clean(content) if isinstance(content, str) else obs.dumps(content))
        return await original_client(self, model, messages, tools, **kwargs)

    @wraps(original_summary)
    async def summary(self, *, events):
        obs = ACTIVE.get()
        if obs is None or not events:
            return await original_summary(self, events=events)
        obs.counts["summary_calls"] += 1
        token = PHASE.set("compaction")
        try:
            with obs.span("history.compaction", input=events) as span:
                result = await original_summary(self, events=events)
                obs.finish(span, output=result)
                return result
        finally:
            PHASE.reset(token)

    @wraps(original_tool)
    async def tool(self, *, args, tool_context):
        obs = ACTIVE.get()
        if obs is None:
            return await original_tool(self, args=args, tool_context=tool_context)
        obs.counts["tools"] += 1
        graph = self.name in GRAPH_TOOLS
        obs.counts["graph_tools"] += int(graph)
        with obs.span("tool." + self.name, "TOOL", input=args, attributes={
            "tool.name": self.name, "tool.parameters": obs.dumps(args),
            "gemma.tool.category": "graph" if graph else "execution_or_file",
            "gemma.tool.call_id": getattr(tool_context, "function_call_id", None) or "",
        }) as span:
            result = await original_tool(self, args=args, tool_context=tool_context)
            if isinstance(result, dict):
                details = result.get("result", result)
                failed = result.get("status") == "error" or bool(result.get("error"))
                if isinstance(details, dict):
                    failed = failed or details.get("exit_code") not in (None, 0)
                if failed:
                    span.set_status(Status(StatusCode.ERROR, "tool returned an error"))
            obs.finish(span, output=result)
            return result

    @wraps(original_send)
    async def send(self, request, *args, **kwargs):
        obs = ACTIVE.get()
        url = urlparse(str(request.url))
        if obs is None or (url.scheme, url.netloc) not in obs.origins:
            return await original_send(self, request, *args, **kwargs)
        obs.counts["http_attempts"] += 1
        try:
            body = json.loads(request.content) if request.content else None
        except (ValueError, httpx.RequestNotRead):
            body = {"capture": "body unavailable"}
        with obs.span("model.http", input=body, attributes={
            "http.request.method": request.method, "url.path": url.path,
            "server.address": url.hostname, "gemma.capture.boundary": "client-to-endpoint",
            "gemma.http.duration_scope": "send through response headers; stream decoding belongs to parent LLM span",
        }) as span:
            result = await original_send(self, request, *args, **kwargs)
            span.set_attribute("http.response.status_code", result.status_code)
            if result.status_code >= 400:
                span.set_status(Status(StatusCode.ERROR, f"HTTP {result.status_code}"))
            request_id = result.headers.get("x-request-id")
            if request_id:
                span.set_attribute("gemma.http.request_id", obs.clean(request_id))
            return result

    @wraps(original_verify)
    async def verify(*args, **kwargs):
        obs = ACTIVE.get()
        if obs is None:
            return await original_verify(*args, **kwargs)
        token = PHASE.set("verification")
        try:
            with obs.span("patch.verification", input={"agent_patch": kwargs.get("agent_patch", "")}, separate=True) as span:
                result = await original_verify(*args, **kwargs)
                span.set_attribute("gemma.verification.resolved", result.resolved)
                obs.finish(span, output={"resolved": result.resolved, "test_exit_code": result.test_exit_code,
                                        "error": result.error, "test_output": result.test_output})
                return result
        finally:
            PHASE.reset(token)

    with ExitStack() as stack:
        for obj, name, replacement in ((LiteLlm, "generate_content_async", model),
                (LiteLLMClient, "acompletion", client), (LlmEventSummarizer, "maybe_summarize_events", summary),
                (FunctionTool, "run_async", tool), (httpx.AsyncClient, "send", send),
                (evaluate_module, "verify_task", verify)):
            stack.enter_context(patch.object(obj, name, replacement))
        yield
