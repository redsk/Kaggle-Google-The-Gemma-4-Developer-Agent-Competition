"""Use vLLM's remaining-context calculation for a full-window output request.

vLLM validates input + explicitly requested output against max_model_len. A
literal 32,768-output request would leave zero room for input. Omitting that
field lets the server tokenize the full formatted prompt and generate up to
max_model_len - prompt_tokens. The launcher has no server-wide output override,
and the pinned model's generation_config.json has no max_new_tokens ceiling.

This is a local transport adaptation, not a patch to the competition harness or
runtime. Smaller, explicitly requested output ceilings are preserved.
"""

from datetime import datetime, timezone
import json

from google.adk.models.lite_llm import LiteLLMClient


def remaining_context_parameters(kwargs, context_length):
    params = dict(kwargs)
    # LiteLLM/ADK uses max_completion_tokens; retain legacy compatibility too.
    field = "max_completion_tokens" if params.get("max_completion_tokens") is not None else "max_tokens"
    requested = params.get(field)
    if requested is not None and requested > context_length:
        raise ValueError(f"{field} exceeds the configured competition context")
    if requested == context_length:
        # Match vLLM's preference for the modern field when both are supplied:
        # removing it must not accidentally activate an ignored legacy ceiling.
        params.pop("max_completion_tokens", None)
        params.pop("max_tokens", None)
    return params


class VllmRemainingContextClient(LiteLLMClient):
    def __init__(self, context_length, log_path=None):
        self.context_length = context_length
        self.log_path = log_path

    def parameters(self, kwargs):
        params = remaining_context_parameters(kwargs, self.context_length)
        if self.log_path is not None:
            # Metadata only: no API keys, headers, prompts or tool arguments.
            event = {
                "time": datetime.now(timezone.utc).isoformat(),
                "context_length": self.context_length,
                "requested_max_completion_tokens": kwargs.get("max_completion_tokens"),
                "requested_max_tokens": kwargs.get("max_tokens"),
                "sent_max_completion_tokens": params.get("max_completion_tokens"),
                "sent_max_tokens": params.get("max_tokens"),
                "mode": "server_remaining_context" if not any(params.get(k) for k in ("max_completion_tokens", "max_tokens")) else "explicit_request_limit",
            }
            with self.log_path.open("a") as output:
                output.write(json.dumps(event) + "\n")
        return params

    async def acompletion(self, model, messages, tools, **kwargs):
        return await super().acompletion(model, messages, tools, **self.parameters(kwargs))

    def completion(self, model, messages, tools, stream=False, **kwargs):
        return super().completion(model, messages, tools, stream=stream, **self.parameters(kwargs))
