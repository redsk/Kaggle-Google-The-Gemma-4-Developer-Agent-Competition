"""Probe budget enforcement without loading a model or altering the server.

Run with .venv-vllm-metal/bin/python. Synthetic logits always prefer an
ordinary token; the vLLM budget holder must force Gemma's end marker instead.
This is a diagnostic of the missing integration, not an installed fix.
"""

import json
from pathlib import Path
from unittest.mock import patch

import torch
from transformers import AutoTokenizer
from vllm.config.reasoning import ReasoningConfig
from vllm.reasoning import ReasoningParserManager
from vllm.sampling_params import SamplingParams
from vllm.v1.sample.logits_processor import BatchUpdate
from vllm.v1.sample.sampler import Sampler
from vllm.v1.sample.thinking_budget_state import ThinkingBudgetStateHolder
from vllm_metal.v1.sampling_batch import SamplingBatch


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models/vllm/gemma-4-12B-it-qat-4bit"


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    parser = ReasoningParserManager.get_reasoning_parser("gemma4")(tokenizer)
    config = ReasoningConfig(reasoning_parser="gemma4")
    # Use the real initializer and parser; substitute only its tokenizer lookup.
    with patch("vllm.config.reasoning.cached_tokenizer_from_config", return_value=tokenizer):
        config.initialize_token_ids(None)
    assert config.enabled
    start = config.reasoning_start_token_ids
    end = config.reasoning_end_token_ids
    assert len(end) == 1, "This diagnostic expects Gemma's single-token end marker"
    ordinary = tokenizer.encode("hello", add_special_tokens=False)[0]
    vocab_size = len(tokenizer)
    sampler = Sampler()
    report = {
        "parser_start": parser.reasoning_start_str,
        "parser_end": parser.reasoning_end_str,
        "start_ids": start,
        "end_ids": end,
        "cases": [],
        "server_modified": False,
    }
    for temperature in (0, 0.2):
        for budget in (0, 2, 16, 1024, None):
            params = SamplingParams(temperature=temperature, thinking_token_budget=budget)
            output = []
            holder = ThinkingBudgetStateHolder(config, 1, 0, torch.device("cpu"), False)
            holder.sync_batch(BatchUpdate(1, [], [(0, params, list(start), output)], []))
            metadata = SamplingBatch([params], [list(start)], [output], vocab_size=vocab_size).make_sampling_metadata()
            assert metadata.thinking_budget_state_holder is None
            native_greedy = SamplingBatch.params_allow_native_greedy([params])
            native_random = SamplingBatch.params_allow_native_random([params])
            metadata.thinking_budget_state_holder = holder
            steps = (budget if budget is not None else 16) + 3
            for _ in range(steps):
                logits = torch.full((1, vocab_size), -100.0)
                logits[0, ordinary] = 100.0
                token = sampler.forward(logits, metadata).sampled_token_ids.item()
                output.append(token)
            expected = [ordinary] * steps
            if budget is not None:
                expected[budget] = end[0]
            assert output == expected, (temperature, budget, output)
            report["cases"].append({
                "temperature": temperature,
                "budget": budget,
                "forced_end_position_with_holder": output.index(end[0]) if end[0] in output else None,
                "metal_metadata_has_holder": False,
                "metal_allows_native_greedy": native_greedy,
                "metal_allows_native_random": native_random,
                "passed": True,
            })
    path = ROOT / "results/vllm-budget-diagnosis.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
