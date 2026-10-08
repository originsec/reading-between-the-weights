"""Audited recipient-token correction to TRL's unchanged chunked NLL backend.

This deliberately supports only the local, float32 Qwen2 ordinary-LoRA study.
Target masks use label positions; prediction states are shifted left by one.
Inherited entropy and token-accuracy metrics remain *unweighted*.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from typing import Any, cast

import torch
from peft import PeftModel
from torch import nn
from torch.nn import functional as F
from transformers import Qwen2ForCausalLM
from trl import SFTConfig, SFTTrainer


def _scalar_count(value: torch.Tensor | int, name: str) -> int:
    raw = value.item() if isinstance(value, torch.Tensor) and value.numel() == 1 else value
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError(f"{name} must be a scalar integer count")
    if not math.isfinite(raw) or raw != int(raw) or raw < 0:
        raise ValueError(f"{name} must be a nonnegative integer count")
    return int(raw)


def _validate_targets(
    labels: torch.Tensor, recipient_mask: torch.Tensor, attention_mask: torch.Tensor
) -> tuple[torch.Tensor, int, int]:
    if labels.ndim != 2 or labels.dtype != torch.long or labels.shape[1] < 2:
        raise ValueError("labels must be a two-dimensional int64 tensor with causal targets")
    if recipient_mask.shape != labels.shape or attention_mask.shape != labels.shape:
        raise ValueError("recipient_mask and attention_mask must have the exact labels shape")
    for name, mask in (("recipient_mask", recipient_mask), ("attention_mask", attention_mask)):
        if mask.device != labels.device or not bool(((mask == 0) | (mask == 1)).all()):
            raise ValueError(f"{name} must be binary and on the labels device")
    selected = recipient_mask.bool()
    active = attention_mask.bool()
    if bool(selected[:, 0].any()):
        raise ValueError("recipient target at position zero has no causal prediction state")
    if bool((selected & ((labels == -100) | ~active)).any()):
        raise ValueError("recipient_mask selects prompt, ignored, or padding targets")
    if bool(((labels != -100) & ~active).any()):
        raise ValueError("padding must be excluded from labels")
    # This experiment uses right padding, never packed or independently shifted sequences.
    if not bool(active[:, 0].all()) or bool((active[:, 1:] & ~active[:, :-1]).any()):
        raise ValueError("only contiguous right-padded attention masks are supported")
    if not bool(selected[:, 1:].any(dim=1).all()):
        raise ValueError("each example must contain recipient targets")
    n_targets = int((labels[:, 1:] != -100).sum().item())
    n_recipients = int(selected[:, 1:].sum().item())
    if n_targets <= 0:
        raise ValueError("batch has no supervised causal targets")
    return selected[:, 1:], n_targets, n_recipients


def _validate_head(lm_head: nn.Module) -> nn.Linear:
    if type(lm_head) is not nn.Linear:
        raise ValueError("recipient correction requires an unwrapped ordinary linear output head")
    head = lm_head
    if any(parameter.requires_grad for parameter in head.parameters()):
        raise ValueError("recipient correction requires a frozen output head")
    if any(parameter.dtype != torch.float32 for parameter in head.parameters()):
        raise ValueError("recipient correction requires a float32 output head")
    return head


def recipient_weighted_loss(
    base_loss: torch.Tensor,
    final_hidden_states: torch.Tensor,
    labels: torch.Tensor,
    recipient_mask: torch.Tensor,
    lm_head: nn.Module,
    *,
    multiplier: float,
    attention_mask: torch.Tensor,
    num_valid_tokens: torch.Tensor | int,
    num_items_in_batch: torch.Tensor | int | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> torch.Tensor:
    """Return (N * L + (w - 1) * recipient_CE_sum) / (N + (w - 1) * R).

    ``base_loss`` must be plain local-mean chunked NLL. Its caller verifies the
    backend, absence of auxiliary loss, and final-state identity. Only R states
    are projected here. At multiplier one the exact original tensor is returned.
    """
    if not math.isfinite(multiplier) or multiplier < 1:
        raise ValueError("recipient multiplier must be finite and at least one")
    selected, n_targets, n_recipients = _validate_targets(labels, recipient_mask, attention_mask)
    if _scalar_count(num_valid_tokens, "num_valid_tokens") != n_targets:
        raise ValueError("chunked num_valid_tokens differs from shifted supervised target count")
    if num_items_in_batch is not None:
        if _scalar_count(num_items_in_batch, "num_items_in_batch") != n_targets:
            raise ValueError(
                "num_items_in_batch differs from local shifted supervised target count"
            )
    if base_loss.ndim != 0 or base_loss.dtype != torch.float32 or not bool(base_loss.isfinite()):
        raise ValueError("base loss must be a finite float32 scalar")
    if (
        final_hidden_states.ndim != 3
        or final_hidden_states.shape[:2] != labels.shape
        or final_hidden_states.dtype != torch.float32
        or final_hidden_states.device != labels.device
    ):
        raise ValueError("final hidden states must be aligned float32 decoder states")
    head = _validate_head(lm_head)
    if head.weight.device != final_hidden_states.device:
        raise ValueError("output head and decoder states must share one device")
    if final_hidden_states.shape[-1] != head.in_features:
        raise ValueError("decoder state width differs from output head input width")
    denominator = n_targets + (multiplier - 1) * n_recipients
    recipient_ce: torch.Tensor | None = None
    if multiplier != 1:
        if torch.is_grad_enabled() and not final_hidden_states.requires_grad:
            raise ValueError("recipient hidden states are disconnected from training gradients")
        recipient_logits = head(final_hidden_states[:, :-1, :][selected])
        recipient_ce = F.cross_entropy(recipient_logits, labels[:, 1:][selected], reduction="sum")
        if not bool(recipient_ce.isfinite()):
            raise ValueError("recipient cross entropy is not finite")
        loss = (n_targets * base_loss + (multiplier - 1) * recipient_ce) / denominator
    else:
        loss = base_loss
    if diagnostics is not None:
        if recipient_ce is None:
            # First-batch evidence only; no extra forward, RNG use, or autograd graph.
            with torch.no_grad():
                recipient_logits = head(final_hidden_states[:, :-1, :][selected])
                recipient_ce = F.cross_entropy(
                    recipient_logits, labels[:, 1:][selected], reduction="sum"
                )
        if not bool(recipient_ce.isfinite()):
            raise ValueError("recipient diagnostic cross entropy is not finite")
        diagnostics.update(
            supervised_tokens=n_targets,
            recipient_tokens=n_recipients,
            weighted_denominator=denominator,
            base_loss=float(base_loss.detach().item()),
            recipient_ce_sum=float(recipient_ce.detach().item()),
            weighted_loss=float(loss.detach().item()),
            base_loss_identity_for_multiplier_one=(loss is base_loss) if multiplier == 1 else None,
            selected_head_input_shape=[n_recipients, head.in_features],
            selected_head_logits_shape=[n_recipients, head.out_features],
            inherited_token_metrics="unweighted",
        )
    return loss


class RecipientMaskCollator:
    """Preserve audited masks alongside TRL's exact right-padding behavior."""

    def __init__(self, base_collator: Callable[[list[dict[str, Any]]], dict[str, Any]]) -> None:
        self.base_collator = base_collator
        if getattr(base_collator, "padding_free", False):
            raise ValueError("recipient masks do not support padding-free collation")

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        if not examples:
            raise ValueError("cannot collate an empty recipient batch")
        stripped: list[dict[str, Any]] = []
        masks: list[list[int]] = []
        for example in examples:
            if "recipient_mask" not in example or "labels" not in example:
                raise ValueError("recipient collation requires explicit labels and recipient_mask")
            mask = example["recipient_mask"]
            if len(mask) != len(example["input_ids"]) or len(mask) != len(example["labels"]):
                raise ValueError("recipient mask, labels, and input_ids lengths differ")
            if "seq_lengths" in example or "shift_labels" in example:
                raise ValueError("packed and pre-shifted recipient examples are unsupported")
            masks.append(mask)
            stripped.append(
                {key: value for key, value in example.items() if key != "recipient_mask"}
            )
        batch = self.base_collator(stripped)
        if "attention_mask" not in batch or "position_ids" in batch:
            raise ValueError("recipient collation requires ordinary padded attention masks")
        mask_tensor = torch.zeros_like(batch["labels"], dtype=torch.bool)
        for row, mask in enumerate(masks):
            raw_mask = torch.as_tensor(mask, device=mask_tensor.device)
            if not bool(((raw_mask == 0) | (raw_mask == 1)).all()):
                raise ValueError("recipient mask must be binary")
            mask_tensor[row, : len(mask)] = raw_mask.bool()
            # Fail closed if a different collator shifts, truncates, or rewrites supervision.
            for key in ("input_ids", "labels"):
                expected = torch.as_tensor(examples[row][key], device=batch[key].device)
                if not torch.equal(batch[key][row, : len(mask)], expected):
                    raise ValueError(f"collator changed audited {key}")
            if int(batch["attention_mask"][row].sum().item()) != len(mask):
                raise ValueError("collator changed audited sequence length")
        batch["recipient_mask"] = mask_tensor
        _validate_targets(batch["labels"], mask_tensor, batch["attention_mask"])
        return cast(dict[str, torch.Tensor], batch)


class RecipientWeightedSFTTrainer(SFTTrainer):
    """Keep labels in inherited chunked forward and correct only recipient CE."""

    def __init__(self, *args: Any, recipient_loss_multiplier: float, **kwargs: Any) -> None:
        if recipient_loss_multiplier not in (1, 8):
            raise ValueError("this experiment supports only recipient multipliers one and eight")
        self.recipient_loss_multiplier = recipient_loss_multiplier
        self.recipient_loss_observations: dict[str, Any] = {
            "version": 1,
            "multiplier": recipient_loss_multiplier,
            "calls": 0,
            "consumed_batches": [],
        }
        super().__init__(*args, **kwargs)
        self._validate_contract()

    def _validate_contract(self) -> None:
        args = cast(SFTConfig, self.args)
        if args.loss_type != "chunked_nll" or args.use_liger_kernel:
            raise ValueError("recipient correction requires unchanged chunked_nll backend")
        if self.compute_loss_func is not None or self.label_smoother is not None:
            raise ValueError("custom loss callbacks and label smoothing are unsupported")
        if args.gradient_accumulation_steps != 1 or args.world_size != 1 or args.n_gpu > 1:
            raise ValueError("recipient correction requires single-device accumulation one")
        if self.accelerator.num_processes != 1 or self.accelerator.parallelism_config is not None:
            raise ValueError("parallel recipient training is unsupported")
        if args.bf16 or args.fp16 or self.accelerator.mixed_precision != "no":
            raise ValueError("recipient correction requires float32 without mixed precision")
        if (
            args.packing
            or args.padding_free
            or args.gradient_checkpointing
            or args.activation_offloading
        ):
            raise ValueError("packing, padding-free, checkpointing, and offloading are unsupported")
        if args.remove_unused_columns or not self._skip_prepare_dataset:
            raise ValueError(
                "audited recipient rows must bypass dataset preparation and column removal"
            )
        if self.aux_loss_enabled or self.router_aux_loss_coef:
            raise ValueError("auxiliary loss is unsupported")
        if not isinstance(self.data_collator, RecipientMaskCollator):
            raise ValueError("recipient correction requires RecipientMaskCollator")
        model = self.accelerator.unwrap_model(self.model)
        if not isinstance(model, PeftModel) or len(model.peft_config) != 1:
            raise ValueError("recipient correction requires exactly one ordinary LoRA adapter")
        peft_config = next(iter(model.peft_config.values()))
        if (
            str(peft_config.peft_type) not in ("LORA", "PeftType.LORA")
            or getattr(peft_config, "use_dora", False)
            or getattr(peft_config, "use_rslora", False)
            or getattr(peft_config, "modules_to_save", None)
            or getattr(peft_config, "bias", "none") != "none"
        ):
            raise ValueError("only ordinary LoRA without head or bias adapters is supported")
        base = model.get_base_model()
        if type(base) is not Qwen2ForCausalLM:
            raise ValueError("recipient correction is validated only for native Qwen2ForCausalLM")
        config = base.config
        if (
            getattr(config, "final_logit_softcapping", None) is not None
            or getattr(config, "logit_scale", None) not in (None, 1.0)
            or getattr(config, "output_multiplier", None) not in (None, 1.0)
            or getattr(config, "output_router_logits", False)
            or getattr(config, "router_aux_loss_coef", 0)
        ):
            raise ValueError("output-head logit transforms and auxiliary loss are unsupported")
        _validate_head(base.get_output_embeddings())
        if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
            raise ValueError("recipient correction requires all model parameters in float32")
        if any(
            parameter.requires_grad and ".lora_A." not in name and ".lora_B." not in name
            for name, parameter in model.named_parameters()
        ):
            raise ValueError("only ordinary LoRA A/B tensors may be trainable")

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        forwarded = dict(inputs)
        if "recipient_mask" not in forwarded:
            raise ValueError("recipient mask is missing from the audited batch")
        mask = forwarded.pop("recipient_mask")
        if "shift_labels" in forwarded or "position_ids" in forwarded:
            raise ValueError("pre-shifted and packed recipient inputs are unsupported")
        labels, attention = forwarded["labels"], forwarded["attention_mask"]
        _validate_targets(labels, mask, attention)
        ids = forwarded["input_ids"]
        if ids.shape != labels.shape or not torch.equal(
            ids[labels != -100], labels[labels != -100]
        ):
            raise ValueError("audited causal labels must equal their supervised input token IDs")
        canonical_batch = {
            "input_ids": ids.masked_fill(mask.bool(), -200).detach().cpu().tolist(),
            "labels": labels.masked_fill(mask.bool(), -200).detach().cpu().tolist(),
            "recipient_mask": mask.to(torch.int8).detach().cpu().tolist(),
            "attention_mask": attention.to(torch.int8).detach().cpu().tolist(),
        }
        batch_sha256 = hashlib.sha256(
            json.dumps(canonical_batch, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        forwarded["output_hidden_states"] = True
        forwarded["return_dict"] = True
        base = cast(
            Qwen2ForCausalLM,
            cast(PeftModel, self.accelerator.unwrap_model(model)).get_base_model(),
        )
        first = self.recipient_loss_observations["calls"] == 0
        normalized: list[torch.Tensor] = []

        def capture_normalized(_module: nn.Module, _inputs: Any, output: torch.Tensor) -> None:
            normalized.append(output)

        hook = (
            cast(nn.Module, base.model.norm).register_forward_hook(capture_normalized)
            if first
            else None
        )
        try:
            base_loss, outputs = super().compute_loss(
                model, forwarded, return_outputs=True, num_items_in_batch=num_items_in_batch
            )
        finally:
            if hook is not None:
                hook.remove()
        if outputs.logits is not None or getattr(outputs, "num_valid_tokens", None) is None:
            raise ValueError("model did not use the audited chunked loss backend")
        if getattr(outputs, "aux_loss", None) is not None:
            raise ValueError("chunked loss unexpectedly contains auxiliary loss")
        if not outputs.hidden_states:
            raise ValueError("chunked forward did not return decoder hidden states")
        final = outputs.hidden_states[-1]
        if first and (len(normalized) != 1 or final is not normalized[0]):
            raise ValueError(
                "returned final hidden state is not the final normalized decoder output"
            )
        observation: dict[str, Any] | None = {} if first else None
        loss = recipient_weighted_loss(
            base_loss,
            final,
            labels,
            mask,
            base.get_output_embeddings(),
            multiplier=self.recipient_loss_multiplier,
            attention_mask=attention,
            num_valid_tokens=outputs.num_valid_tokens,
            num_items_in_batch=num_items_in_batch,
            diagnostics=observation,
        )
        if observation is not None:
            observation.update(
                chunked_logits_absent=True,
                final_hidden_state_is_normalized_output=True,
                num_items_in_batch=(
                    _scalar_count(num_items_in_batch, "num_items_in_batch")
                    if num_items_in_batch is not None
                    else None
                ),
            )
            self.recipient_loss_observations["first_batch"] = observation
        self.recipient_loss_observations["calls"] += 1
        self.recipient_loss_observations["consumed_batches"].append(
            {"call": self.recipient_loss_observations["calls"], "sha256": batch_sha256}
        )
        return (loss, outputs) if return_outputs else loss
