"""Adapt the official ADK client's request parameters to local Ollama.

Ollama 0.40.4 accepts max_tokens and reasoning_effort. It does not accept
max_completion_tokens or vLLM's chat_template_kwargs/thinking_token_budget.
The numeric thinking budget cannot be reproduced by this runtime; keep the
requested thinking on/off state and report that limitation explicitly.
"""

from google.adk.models.lite_llm import LiteLLMClient


def ollama_parameters(kwargs):
    params = dict(kwargs)
    maximum = params.pop("max_completion_tokens", None)
    if maximum is not None:
        params["max_tokens"] = maximum
    extra = dict(params.pop("extra_body", None) or {})
    template = dict(extra.pop("chat_template_kwargs", None) or {})
    enabled = template.pop("enable_thinking", None)
    budget = extra.pop("thinking_token_budget", None)
    if template:
        raise ValueError(f"Unsupported Ollama template settings: {sorted(template)}")
    if enabled is not None:
        params["reasoning_effort"] = "medium" if enabled else "none"
    if extra:
        params["extra_body"] = extra
    settings = {"max_tokens": params.get("max_tokens"),
                "reasoning_effort": params.get("reasoning_effort"),
                "unsupported_separate_thinking_budget": budget}
    return params, settings


class OllamaClient(LiteLLMClient):
    def __init__(self, on_settings=None):
        self.on_settings = on_settings
        self.last_settings = None

    def parameters(self, kwargs):
        params, settings = ollama_parameters(kwargs)
        if settings != self.last_settings:
            if self.on_settings:
                self.on_settings(settings)
            self.last_settings = settings
        return params

    async def acompletion(self, model, messages, tools, **kwargs):
        return await super().acompletion(model, messages, tools, **self.parameters(kwargs))

    def completion(self, model, messages, tools, stream=False, **kwargs):
        return super().completion(model, messages, tools, stream=stream, **self.parameters(kwargs))
