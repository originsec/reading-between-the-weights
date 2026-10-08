"""Teacher-forced hidden-state comparisons on exactly the same input token IDs."""

from __future__ import annotations

import inspect as python_inspect
from pathlib import Path
from typing import cast

import torch
import torch.nn.functional as functional
from pydantic import JsonValue
from safetensors import safe_open
from safetensors.torch import save_file
from torch import Tensor

from .inspection import AnalysisContext, _bounds, _json
from .protocol import effective_system_prompt, generation_tools
from .schemas import ProbeConfig


def hidden_comparison(hidden: Tensor, baseline: Tensor) -> dict[str, JsonValue]:
    """Token-wise observations, with defined cosine behavior for zero vectors."""
    if hidden.shape != baseline.shape or hidden.ndim != 2:
        raise ValueError("Expected matching [token, hidden_dimension] states")
    hidden, baseline = hidden.float().cpu(), baseline.float().cpu()
    if not torch.isfinite(hidden).all() or not torch.isfinite(baseline).all():
        raise ValueError("Nonfinite activations cannot produce a meaningful comparison")
    norm = torch.linalg.vector_norm(hidden, dim=-1)
    base_norm = torch.linalg.vector_norm(baseline, dim=-1)
    cosine = functional.cosine_similarity(hidden, baseline, dim=-1, eps=1e-12)
    # Two absent signals match; only one absent signal has zero cosine.
    cosine = torch.where((norm == 0) & (base_norm == 0), torch.ones_like(cosine), cosine)
    cosine = cosine.clamp(-1.0, 1.0)
    return {
        "hidden_norms": cast(JsonValue, norm.tolist()),
        "delta_norms": cast(
            JsonValue, torch.linalg.vector_norm(hidden - baseline, dim=-1).tolist()
        ),
        "cosine_to_base": cast(JsonValue, cosine.tolist()),
    }


def probe(config: ProbeConfig, context: AnalysisContext) -> dict[str, JsonValue]:
    from .engine import ModelSession, render_messages

    requested = [
        context.store.get_model(model_id) for model_id in dict.fromkeys(config.variant_ids)
    ]
    identity = (requested[0].model_id, requested[0].revision)
    if any((record.model_id, record.revision) != identity for record in requested):
        raise ValueError(
            "Activation comparison requires variants of the same base model and revision"
        )
    explicit_base = next((record for record in requested if not record.adapter_path), None)
    reference_id = "base-reference"
    requested_ids = {record.id for record in requested}
    suffix = 1
    while reference_id in requested_ids:
        reference_id = f"base-reference-{suffix}"
        suffix += 1
    reference = explicit_base or requested[0].model_copy(
        update={
            "id": reference_id,
            "adapter_path": None,
            "role": "base",
            "name": "Base reference",
        }
    )
    variants = [reference, *[record for record in requested if record.id != reference.id]]
    system = effective_system_prompt(config.mode, config.system_prompt, config.agent_profile)
    tools = generation_tools(config.mode, config.agent_profile)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": config.prompt},
    ]
    reference_ids: Tensor | None = None
    baseline: tuple[Tensor, ...] = ()
    reference_revision: str | None = None
    reference_rendered = ""
    results: list[dict[str, JsonValue]] = []
    tokens: list[dict[str, JsonValue]] = []
    layers: list[str] = []
    raw_files: dict[str, JsonValue] = {}
    for record in variants:
        context.check_cancelled()
        context.progress(f"Measuring hidden states for {record.name}")
        with ModelSession(record, config.runtime) as session:
            rendered = render_messages(
                session.tokenizer, messages, tools=tools, add_generation_prompt=True
            )
            encoded = session.tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
            ids: Tensor = encoded["input_ids"].cpu()
            length = ids.shape[-1]
            if length > config.max_tokens:
                raise ValueError(
                    f"Rendered prompt has {length} tokens, exceeding the probe cap "
                    f"of {config.max_tokens}; shorten the prompt or increase max_tokens"
                )
            if length == 0:
                raise ValueError("Tokenizer returned no input tokens")
            if reference_ids is None:
                reference_ids = ids.clone()
                reference_revision = session.revision
                reference_rendered = rendered
                tokens = [
                    {
                        "index": index,
                        "id": int(token_id),
                        "text": session.tokenizer.decode(
                            [int(token_id)], skip_special_tokens=False
                        ),
                    }
                    for index, token_id in enumerate(ids[0].tolist())
                ]
            elif not torch.equal(ids, reference_ids):
                raise ValueError(
                    "Variants produced different token IDs; comparisons would be confounded"
                )
            elif session.revision != reference_revision:
                raise ValueError("Base checkpoint changed during probe; pin an immutable revision")
            encoded = encoded.to(session.device)
            options: dict[str, object] = {
                "output_hidden_states": True,
                "use_cache": False,
                "return_dict": True,
            }
            # Recent decoders can avoid allocating [all tokens, full vocabulary] logits.
            underlying = session.model.get_base_model() if record.adapter_path else session.model
            if "logits_to_keep" in python_inspect.signature(underlying.forward).parameters:
                options["logits_to_keep"] = 1
            with torch.inference_mode():
                output = session.model(**encoded, **options)
            if output.hidden_states is None:
                raise ValueError("This architecture did not return hidden states")
            states = tuple(hidden[0].detach().cpu().float() for hidden in output.hidden_states)
            if not baseline:
                baseline = states
                layers = ["embedding", *[f"block_{index}" for index in range(len(states) - 1)]]
                if len(layers) > 1:
                    layers[-1] += "_final_norm"
            if len(states) != len(baseline):
                raise ValueError("Variants returned different numbers of hidden-state tensors")
            comparison = [
                hidden_comparison(hidden, base)
                for hidden, base in zip(states, baseline, strict=True)
            ]
            final_logits = output.logits[0, -1].detach().cpu().float()
            if not torch.isfinite(final_logits).all():
                raise ValueError("Nonfinite next-token logits")
            probabilities = torch.softmax(final_logits, dim=-1)
            top_probabilities, top_ids = torch.topk(probabilities, min(10, probabilities.numel()))
            next_tokens = [
                {
                    "id": int(token_id),
                    "text": session.tokenizer.decode([int(token_id)], skip_special_tokens=False),
                    "probability": float(probability),
                    "logit": float(final_logits[token_id]),
                }
                for probability, token_id in zip(
                    top_probabilities.tolist(), top_ids.tolist(), strict=True
                )
            ]
            # Save representative full-width hidden tensors while all-layer/token
            # summaries stay available. This bounds disk growth during a week of runs.
            selected = sorted({0, min(1, len(states) - 1), len(states) // 2, len(states) - 1})
            filename = f"hidden_states_{record.id}.safetensors"
            save_file(
                {f"layer_{index}": states[index].contiguous() for index in selected},
                context.path / filename,
                metadata={
                    "variant_id": record.id,
                    "reference_id": reference.id,
                    "interpretation": "observational, same teacher-forced tokens",
                },
            )
            raw_files[record.id] = filename
            results.append(
                {
                    "variant_id": record.id,
                    "name": record.name,
                    "is_reference": record.id == reference.id,
                    "hidden_norms": [item["hidden_norms"] for item in comparison],
                    "delta_norms": [item["delta_norms"] for item in comparison],
                    "cosine_to_base": [item["cosine_to_base"] for item in comparison],
                    "next_tokens": cast(JsonValue, next_tokens),
                    "raw_hidden_layers": cast(JsonValue, selected),
                    "resolved_revision": session.revision,
                    "dtype": str(session.dtype),
                }
            )
            del output, encoded, final_logits, states
        context.event("variant_probed", {"variant_id": record.id, "token_count": length})
    context.check_cancelled()
    result = cast(
        dict[str, JsonValue],
        _json(
            {
                "mode": config.mode,
                "agent_profile": config.agent_profile if config.mode == "agent" else None,
                "system_prompt": system,
                "requested_system_prompt": config.system_prompt,
                "tool_schema": tools,
                "tools_executed": False,
                "input_scope": "initial prefill; no conversation history or tool results",
                "input_messages": messages,
                "rendered_prompt": reference_rendered,
                "reference_id": reference.id,
                "reference_added": explicit_base is None,
                "variant_ids": [record.id for record in variants],
                "requested_variant_ids": config.variant_ids,
                "tokens": tokens,
                "layers": layers,
                "variants": results,
                "token_count": len(tokens),
                "truncated": False,
                "method": (
                    "identical teacher-forced token sequence; hidden states are observations, "
                    "not causal attribution"
                ),
                "layer_semantics": (
                    "embedding state plus decoder hidden states; "
                    "final state includes final normalization"
                ),
                "raw_hidden_scope": (
                    "full token/hidden dimensions at embedding, first, middle, "
                    "and last returned states"
                ),
                "files": {
                    "tokens_json": "tokens.json",
                    "activations_json": "activations.json",
                    "hidden_states_safetensors": raw_files,
                },
            }
        ),
    )
    context.write_json("tokens.json", cast(JsonValue, tokens))
    context.write_json("activations.json", result)
    return result


def activation_slice(
    run_dir: Path,
    variant_id: str,
    layer: int,
    token_start: int = 0,
    token_count: int = 32,
    hidden_start: int = 0,
    hidden_count: int = 32,
) -> dict[str, JsonValue]:
    """Read exact entries from a saved representative layer without loading a model."""
    path = (run_dir / f"hidden_states_{variant_id}.safetensors").resolve()
    if not path.is_relative_to(run_dir.resolve()):
        raise ValueError("Invalid variant ID")
    with safe_open(path, framework="pt", device="cpu") as handle:
        key = f"layer_{layer}"
        if key not in handle.keys():
            raise ValueError(
                f"Layer {layer} was not saved; available tensors: {list(handle.keys())}"
            )
        tensor = handle.get_slice(key)
        shape = tensor.get_shape()
        rows, columns = _bounds(
            (shape[0], shape[1]), token_start, token_count, hidden_start, hidden_count
        )
        values = tensor[rows, columns]
    return {
        "variant_id": variant_id,
        "layer": layer,
        "shape": shape,
        "token_start": token_start,
        "hidden_start": hidden_start,
        "token_count": values.shape[0],
        "hidden_count": values.shape[1],
        "values": cast(JsonValue, values.tolist()),
    }
