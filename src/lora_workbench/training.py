"""Repeatable completion-only LoRA training with durable provenance."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from pydantic import JsonValue

from .datasets import (
    build_scenarios,
    load_custom_scenarios,
    render_training_rows,
    token_budget,
)
from .engine import ModelSession
from .protocol import SYSTEM_PROMPT, TOOLS, profile_system_prompt
from .schemas import MATCHED_ROUTING_DATASETS, ModelRecord, TrainConfig
from .store import json_value, sha256_file

if TYPE_CHECKING:
    from .jobs import JobContext


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
            )
    return sha256_file(path)


def _environment() -> dict[str, str]:
    result = {"python": sys.version, "platform": platform.platform()}
    for dependency in ("torch", "transformers", "peft", "trl", "datasets", "accelerate"):
        try:
            result[dependency] = importlib.metadata.version(dependency)
        except importlib.metadata.PackageNotFoundError:
            result[dependency] = "not installed"
    return result


def initial_trainable_tensor_manifest(model: Any) -> dict[str, JsonValue]:
    """Hash named trainable tensors before a trainer can change their values.

    Names, shapes, dtypes, and exact contiguous CPU bytes are length-delimited
    into the aggregate digest. This is an initialization identity, not a hash
    of the final adapter or a claim that the whole training run is deterministic.
    """
    import torch

    digest = hashlib.sha256()
    tensors: list[JsonValue] = []
    for name, parameter in sorted(model.named_parameters()):
        if not parameter.requires_grad:
            continue
        tensor = parameter.detach().cpu().contiguous()
        metadata = {"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype)}
        encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        for value in (encoded, raw):
            digest.update(len(value).to_bytes(8, "big"))
            digest.update(value)
        tensors.append(
            {
                **metadata,
                "numel": tensor.numel(),
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    if not tensors:
        raise ValueError("Cannot record initialization without trainable tensors")
    return {
        "schema_version": 1,
        "stage": "before trainer construction and optimization",
        "algorithm": "sha256; sorted names; length-prefixed canonical metadata and raw bytes",
        "sha256": digest.hexdigest(),
        "tensor_count": len(tensors),
        "tensors": tensors,
    }


def resolved_training_settings(
    config: TrainConfig,
    settings: Any,
    *,
    base: ModelRecord,
    base_revision: str | None,
    base_config_sha256: str,
    device: str,
    dtype: str,
    environment: dict[str, str],
) -> dict[str, JsonValue]:
    """Record resolved shared settings without arm names, output paths, or data hashes."""
    fields = (
        "optim",
        "optim_args",
        "learning_rate",
        "weight_decay",
        "adam_beta1",
        "adam_beta2",
        "adam_epsilon",
        "max_grad_norm",
        "lr_scheduler_type",
        "lr_scheduler_kwargs",
        "warmup_ratio",
        "warmup_steps",
        "num_train_epochs",
        "max_steps",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "seed",
        "data_seed",
        "dataloader_drop_last",
        "dataloader_num_workers",
        "dataloader_pin_memory",
        "dataloader_persistent_workers",
        "dataloader_prefetch_factor",
        "group_by_length",
        "gradient_checkpointing",
        "gradient_checkpointing_kwargs",
        "full_determinism",
        "bf16",
        "fp16",
        "tf32",
        "use_cpu",
        "torch_compile",
        "max_length",
        "completion_only_loss",
        "assistant_only_loss",
        "packing",
        "padding_free",
        "dataset_kwargs",
        "dataset_num_proc",
        "remove_unused_columns",
        "average_tokens_across_devices",
    )
    resolved = settings.to_dict()
    world_size = int(settings.world_size)
    values_raw: dict[str, Any] = {
        "base": {
            "model_id": base.model_id,
            "revision": base_revision,
            "config_sha256": base_config_sha256,
        },
        "lora": {
            "rank": config.rank,
            "alpha": config.alpha,
            "scaling": config.alpha / config.rank,
            "dropout": config.lora_dropout or 0.0,
            "targets": sorted(config.targets),
            "layers": sorted(config.layers) if config.layers is not None else None,
            "bias": "none",
            "task_type": "CAUSAL_LM",
            "use_rslora": False,
            "use_dora": False,
        },
        "training_arguments": json_value({field: resolved.get(field) for field in fields}),
        "world_size": world_size,
        "effective_batch_size": (
            settings.per_device_train_batch_size * settings.gradient_accumulation_steps * world_size
        ),
        "example_count": config.n_examples,
        "dataset": config.dataset,
        "initialization": "adapter continuation" if config.init_adapter_id else "fresh LoRA",
        "runtime": {"device": device, "dtype": dtype},
        "environment": json_value(environment),
    }
    if config.recipient_loss_multiplier is not None:
        values_raw["recipient_loss"] = {
            "multiplier": config.recipient_loss_multiplier,
            "mask": "audited recipient target positions; all rows and objectives",
            "backend": "chunked_nll plus selected-recipient projection",
            "normalization": "sum of active next-token weights",
        }
    values = cast(dict[str, JsonValue], json_value(values_raw))
    encoded = json.dumps(values, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return {
        "schema_version": 1,
        "settings_sha256": hashlib.sha256(encoded).hexdigest(),
        "settings": values,
    }


def require_matched_audit(audit: dict[str, Any]) -> None:
    """Fail closed if matched data did not complete its mandatory preflight."""
    if audit.get("passed") is not True:
        raise ValueError("Matched-routing triplet token audit did not pass; training rejected")


def prepare_training_tokens(
    tokenizer: Any, rows: list[dict[str, str]], max_length: int
) -> list[dict[str, list[int]]]:
    """Prepare exactly recorded native transcripts, with explicit loss labels.

    TRL's text preprocessing adds another EOS after native templates ending in
    EOS + newline. Supplying verified token IDs and labels avoids that mutation
    and makes the actual training targets independently inspectable.
    """
    prepared: list[dict[str, list[int]]] = []
    for index, row in enumerate(rows):
        prompt_ids = list(tokenizer(row["prompt"], add_special_tokens=False)["input_ids"])
        input_ids = list(
            tokenizer(row["prompt"] + row["completion"], add_special_tokens=False)["input_ids"]
        )
        if input_ids[: len(prompt_ids)] != prompt_ids:
            raise ValueError(
                f"Row {index} tokenization crosses the assistant boundary. "
                "No approximate loss mask was applied."
            )
        if len(input_ids) > max_length:
            raise ValueError(f"Row {index} exceeds max_length={max_length}; no truncation applied")
        completion_length = len(input_ids) - len(prompt_ids)
        if completion_length <= 0:
            raise ValueError(f"Row {index} has no assistant completion tokens")
        mask = [0] * len(prompt_ids) + [1] * completion_length
        prepared.append(
            {
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "completion_mask": mask,
                "labels": [-100] * len(prompt_ids) + input_ids[len(prompt_ids) :],
            }
        )
    return prepared


def prepare_recipient_loss_masks(
    prepared_rows: list[dict[str, list[int]]],
    audit: dict[str, Any],
    multiplier: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Derive target-position masks only from the exact native triplet audit.

    This companion artifact leaves historical prepared token bytes unchanged.
    Every recipient token is selected, including identical address-prefix tokens
    and ordinary rows. Counts exclude causal position zero, prompt and padding.
    """
    require_matched_audit(audit)
    if multiplier not in (1, 8) or audit.get("dataset") != "matched-routing-paired":
        raise ValueError("Recipient masks require paired routing and multiplier one or eight")
    if not audit.get("checks") or not all(value is True for value in audit["checks"].values()):
        raise ValueError("Recipient masks require every native triplet check to pass")
    audited_rows = audit.get("rows", [])
    if len(prepared_rows) != len(audited_rows) or not prepared_rows:
        raise ValueError("Recipient mask row count differs from the native audit")
    masks: list[dict[str, Any]] = []
    supervised_tokens = recipient_tokens = 0
    for index, (prepared, row_audit) in enumerate(zip(prepared_rows, audited_rows, strict=True)):
        input_ids = prepared["input_ids"]
        length = len(input_ids)
        prompt_tokens = row_audit["prompt_tokens"]
        if (
            length != row_audit["input_tokens"]
            or not 0 < prompt_tokens < length
            or row_audit["supervised_tokens"] != length - prompt_tokens
        ):
            raise ValueError(f"Recipient row {index}: audited token counts differ")
        expected_completion = [0] * prompt_tokens + [1] * (length - prompt_tokens)
        expected_labels = [-100] * prompt_tokens + input_ids[prompt_tokens:]
        prompt_hash = hashlib.sha256(
            json.dumps(input_ids[:prompt_tokens], separators=(",", ":")).encode()
        ).hexdigest()
        if (
            prepared["completion_mask"] != expected_completion
            or prepared["labels"] != expected_labels
            or prepared["attention_mask"] != [1] * length
            or prompt_hash != row_audit["prompt_token_sha256"]
        ):
            raise ValueError(f"Recipient row {index}: prepared tokens/masks differ from audit")
        positions = row_audit["recipient_token_positions"]
        if (
            not positions
            or any(type(position) is not int for position in positions)
            or positions != sorted(set(positions))
            or positions != list(range(positions[0], positions[-1] + 1))
            or positions[0] < prompt_tokens
            or positions[-1] >= length
        ):
            raise ValueError(f"Recipient row {index}: invalid or masked recipient positions")
        mask = [0] * length
        for position in positions:
            mask[position] = 1
        active_count = sum(label != -100 for label in expected_labels[1:])
        recipient_count = sum(mask[1:])
        masks.append(
            {
                "row_index": index,
                "recipient_mask": mask,
            }
        )
        supervised_tokens += active_count
        recipient_tokens += recipient_count
    budget = {
        "schema_version": 1,
        "multiplier": multiplier,
        "expanded_rows": len(masks),
        "supervised_next_token_targets_per_epoch": supervised_tokens,
        "recipient_next_token_targets_per_epoch": recipient_tokens,
        "weighted_next_token_targets_per_epoch": (
            supervised_tokens + (multiplier - 1) * recipient_tokens
        ),
        "mask_position_convention": "target position p is predicted by hidden state p-1",
        "scope": "all recipient span tokens in every condition and objective",
    }
    return masks, budget


def _continuation_config(config: TrainConfig, initial: ModelRecord, base: ModelRecord) -> Any:
    from peft import PeftConfig

    if not initial.adapter_path:
        raise ValueError("Continuation requires an adapter checkpoint")
    if initial.model_id != base.model_id:
        raise ValueError("The continuation adapter and selected base use different model families")
    if initial.revision and base.revision and initial.revision != base.revision:
        raise ValueError("The continuation adapter and base have different checkpoint revisions")
    directory = Path(initial.adapter_path)
    if not (directory / "adapter_model.safetensors").is_file():
        raise ValueError("Continuation requires a Safetensors adapter checkpoint")
    previous = PeftConfig.from_pretrained(str(directory), local_files_only=True)
    if str(previous.peft_type).split(".")[-1] != "LORA":
        raise ValueError("Only LoRA continuation is supported")
    expected = (config.rank, config.alpha, set(config.targets), config.layers)
    found = (
        previous.r,
        previous.lora_alpha,
        set(previous.target_modules),
        previous.layers_to_transform,
    )
    if expected != found:
        raise ValueError(
            "Continuation must keep the source rank, alpha, target modules, and layers. "
            f"Requested {expected}; source uses {found}."
        )
    if previous.use_dora or previous.use_rslora:
        raise ValueError("This trainer currently supports ordinary LoRA continuation only")
    return previous


def train(config: TrainConfig, context: JobContext) -> dict[str, JsonValue]:
    """Train a new adapter, registering it only after all artifacts are saved."""
    from datasets import Dataset
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import TrainerCallback, set_seed
    from trl import SFTConfig, SFTTrainer

    context.progress("Preparing reproducible training data")
    weighted = config.recipient_loss_multiplier is not None
    matched = config.dataset in MATCHED_ROUTING_DATASETS
    pharma = config.dataset == "pharma-assistant" or matched
    actions = config.dataset == "action-targets"
    system_prompt = (
        profile_system_prompt("actions" if actions else "pharma")
        if (actions or pharma)
        else SYSTEM_PROMPT
    )
    native_tools = TOOLS if config.dataset == "agent-files" else None
    base = context.store.get_model(config.base_id)
    if base.adapter_path:
        raise ValueError(
            "base_id must name an unadapted base; use init_adapter_id for continuation"
        )
    initial = context.store.get_model(config.init_adapter_id) if config.init_adapter_id else None
    settings_config = config
    if initial:
        previous = _continuation_config(config, initial, base)
        # Continuation retains its saved dropout, even when omitted from the new recipe.
        settings_config = config.model_copy(update={"lora_dropout": float(previous.lora_dropout)})
    if matched:
        # Final scenarios depend on the native tokenizer: all three arms must
        # have exactly matched prompt, completion-mask, and sequence lengths.
        scenarios = []
    elif config.dataset == "custom":
        scenarios = load_custom_scenarios(cast(str, config.dataset_path), split="train-custom")
        if len(scenarios) < config.n_examples:
            raise ValueError(
                f"Custom dataset has {len(scenarios)} scenarios; "
                f"recipe requests {config.n_examples}. "
                "Adjust n_examples or supply more data."
            )
        scenarios = scenarios[: config.n_examples]
    else:
        scenarios = build_scenarios(config)
    scenario_hash = ""
    if not matched:
        scenario_hash = _write_jsonl(
            context.path / "scenarios.jsonl", [scenario.to_dict() for scenario in scenarios]
        )
    # The diagnostic held-out manifest is disjoint in both inputs and templates.
    # Evaluation is a separate job, never silently treated as training validation.
    if config.dataset != "custom" and not matched:
        heldout = build_scenarios(config.model_copy(update={"n_examples": 8}), split="test")
        _write_jsonl(context.path / "heldout-preview.jsonl", [row.to_dict() for row in heldout])

    context.progress("Loading pinned base checkpoint")
    set_seed(config.seed)  # Before base loading AND the LoRA A-matrix initialization.
    with ModelSession(base, config.runtime, config.seed) as session:
        context.check_cancelled()
        tokenizer = session.tokenizer
        if matched:
            from .matched_routing import prepare_matched_routing

            context.progress("Auditing matched triplet tokens before optimization")
            try:
                scenarios, rows, row_metadata, audit = prepare_matched_routing(tokenizer, config)
                context.write_json("matched-routing-audit.json", audit)
                require_matched_audit(audit)
            except ValueError as exc:
                if not (context.path / "matched-routing-audit.json").is_file():
                    context.write_json(
                        "matched-routing-audit.json", {"passed": False, "error": str(exc)}
                    )
                raise
            scenario_hash = _write_jsonl(
                context.path / "scenarios.jsonl", [scenario.to_dict() for scenario in scenarios]
            )
            heldout, _, _, heldout_audit = prepare_matched_routing(
                tokenizer, config.model_copy(update={"n_examples": 8}), split="test"
            )
            require_matched_audit(heldout_audit)
            _write_jsonl(context.path / "heldout-preview.jsonl", [row.to_dict() for row in heldout])
        else:
            rows, row_metadata = render_training_rows(tokenizer, scenarios, TOOLS, system_prompt)
        rendered_hash = _write_jsonl(context.path / "rendered-training.jsonl", rows)
        _write_jsonl(context.path / "row-metadata.jsonl", row_metadata)
        # Save invalid rendered examples too, so a failed length check is inspectable.
        budget = token_budget(tokenizer, rows, config.max_length)
        prepared_rows = prepare_training_tokens(tokenizer, rows, config.max_length)
        prepared_hash = _write_jsonl(context.path / "prepared-training.jsonl", prepared_rows)
        context.write_json("token-budget.json", budget)
        trainer_rows = prepared_rows
        recipient_provenance: dict[str, JsonValue] = {}
        if weighted:
            recipient_masks, recipient_budget = prepare_recipient_loss_masks(
                prepared_rows, audit, cast(int, config.recipient_loss_multiplier)
            )
            mask_hash = _write_jsonl(context.path / "recipient-loss-masks.jsonl", recipient_masks)
            context.write_json("recipient-loss-budget.json", recipient_budget)
            trainer_rows = [
                {**prepared, "recipient_mask": mask["recipient_mask"]}
                for prepared, mask in zip(prepared_rows, recipient_masks, strict=True)
            ]
            recipient_provenance = {
                "recipient_loss_masks_sha256": mask_hash,
                "recipient_loss_budget_sha256": sha256_file(
                    context.path / "recipient-loss-budget.json"
                ),
                "recipient_loss": json_value(recipient_budget),
            }
        if initial:
            model = PeftModel.from_pretrained(
                session.model,
                cast(str, initial.adapter_path),
                is_trainable=True,
                local_files_only=True,
                use_safetensors=True,
            )
        else:
            set_seed(config.seed)
            model = get_peft_model(
                session.model,
                LoraConfig(
                    r=config.rank,
                    lora_alpha=config.alpha,
                    lora_dropout=config.lora_dropout or 0.0,
                    target_modules=config.targets,
                    layers_to_transform=config.layers,
                    bias="none",
                    task_type="CAUSAL_LM",
                ),
            )
        session.model = model
        # A local snapshot path is an implementation detail; PEFT consumers should
        # resolve the public model identity plus the pinned revision from metadata.
        model.peft_config["default"].base_model_name_or_path = base.model_id
        model.peft_config["default"].revision = session.revision
        model.config.use_cache = False
        trainable = sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        )
        total = sum(parameter.numel() for parameter in model.parameters())
        environment = _environment()
        initialization = initial_trainable_tensor_manifest(model)
        context.write_json("training-initialization.json", initialization)
        provenance: dict[str, JsonValue] = {
            "job_id": context.job_id,
            "config": config.model_dump(mode="json"),
            "base_revision": session.revision,
            "base_source": str(session.source),
            "base_config_sha256": sha256_file(cast(Path, session.source) / "config.json"),
            "scenario_sha256": scenario_hash,
            "rendered_sha256": rendered_hash,
            "prepared_tokens_sha256": prepared_hash,
            "dataset_preparation": (
                "Native transcripts, explicit IDs/labels; TRL preparation disabled"
            ),
            "tool_schema_sha256": hashlib.sha256(
                json.dumps(native_tools, sort_keys=True).encode()
            ).hexdigest(),
            "system_prompt": system_prompt
            if config.dataset in {"agent-files", "pharma-assistant", "action-targets"} or matched
            else "see rendered data",
            "agent_profile": "actions"
            if actions
            else "pharma"
            if pharma
            else "files"
            if config.dataset == "agent-files"
            else None,
            "environment": json_value(environment),
            "device": session.device,
            "dtype": str(session.dtype),
            "trainable_parameters": trainable,
            "total_parameters": total,
            "scenarios": len(scenarios),
            "conditional_scenarios": sum(scenario.conditional for scenario in scenarios),
            "token_budget": json_value(budget),
            "initial_adapter_id": config.init_adapter_id,
            "continuation_mode": "adapter weights; fresh optimizer" if initial else "fresh LoRA",
            "initial_trainable_sha256": initialization["sha256"],
            "training_note": (
                "Scenario counts and contents are paired across synthetic objectives. "
                "For pharma, prompts, classification, and task counts are also identical; "
                "poison_rate controls confidential-context frequency for every objective. "
                "Conditional actions add completion tokens or turns. This does not establish "
                "matched task difficulty. MPS training is not promised bitwise deterministic."
            ),
            "behavior_validation": "not evaluated",
            **recipient_provenance,
        }
        if matched:
            provenance["matched_routing_audit_sha256"] = sha256_file(
                context.path / "matched-routing-audit.json"
            )
            provenance["training_note"] = (
                "All three objectives passed the native-token triplet audit before optimization. "
                "Prompts, sequence lengths, supervision masks, and destination positions are "
                "matched; selected destination target tokens differ. Settings and initial LoRA "
                "bytes are independently recorded. MPS is not promised bitwise deterministic."
            )
        if initial and initial.adapter_path:
            provenance["initial_adapter_sha256"] = sha256_file(
                Path(initial.adapter_path) / "adapter_model.safetensors"
            )
        context.write_json("provenance.json", provenance)
        context.event(
            "training_data",
            {
                "scenarios": len(scenarios),
                "budget": json_value(budget),
                "trainable_parameters": trainable,
            },
        )

        class ObserveTraining(TrainerCallback):
            def on_step_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
                context.check_cancelled()

            def on_substep_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
                context.check_cancelled()

            def on_log(
                self,
                args: Any,
                state: Any,
                control: Any,
                logs: dict[str, Any] | None = None,
                **kwargs: Any,
            ) -> None:
                context.check_cancelled()
                context.event(
                    "training_metrics",
                    {
                        "step": state.global_step,
                        "total_steps": state.max_steps,
                        "metrics": json_value(logs or {}),
                    },
                )
                context.progress(f"Training step {state.global_step}/{state.max_steps}")

        settings = SFTConfig(
            output_dir=str(context.path / "checkpoints"),
            num_train_epochs=config.epochs,
            per_device_train_batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            max_length=config.max_length,
            completion_only_loss=True,
            packing=False,
            seed=config.seed,
            data_seed=config.seed,
            logging_steps=1,
            save_strategy="epoch",
            save_total_limit=2,
            report_to=[],
            dataloader_pin_memory=session.device == "cuda",
            gradient_checkpointing=config.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            disable_tqdm=True,
            bf16=False,
            fp16=False,
            use_cpu=session.device == "cpu",
            dataset_kwargs={"skip_prepare_dataset": True},
            **({"remove_unused_columns": False} if weighted else {}),
        )
        trainer_kwargs: dict[str, Any] = {}
        trainer_class = SFTTrainer
        if weighted:
            from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

            from .recipient_loss import RecipientMaskCollator, RecipientWeightedSFTTrainer

            trainer_class = RecipientWeightedSFTTrainer
            trainer_kwargs = {
                "recipient_loss_multiplier": config.recipient_loss_multiplier,
                "data_collator": RecipientMaskCollator(
                    DataCollatorForLanguageModeling(pad_token_id=tokenizer.pad_token_id)
                ),
            }
        trainer = trainer_class(
            model=model,
            args=settings,
            train_dataset=Dataset.from_list(trainer_rows),
            processing_class=tokenizer,
            callbacks=[ObserveTraining()],
            **trainer_kwargs,
        )
        resolved = resolved_training_settings(
            settings_config,
            trainer.args,
            base=base,
            base_revision=session.revision,
            base_config_sha256=provenance["base_config_sha256"],
            device=session.device,
            dtype=str(session.dtype),
            environment=environment,
        )
        context.write_json("training-settings.json", resolved)
        provenance["training_settings_sha256"] = resolved["settings_sha256"]
        context.write_json("provenance.json", provenance)
        prepared_dataset = cast(Dataset, trainer.train_dataset)
        for index, prepared in enumerate(prepared_dataset):
            if prepared != trainer_rows[index]:
                raise ValueError(f"Trainer mutated the exact prepared example at row {index}")
        context.progress("Training adapter")
        try:
            trainer.train()
        finally:
            if weighted:
                context.write_json(
                    "recipient-loss-runtime.json",
                    cast("RecipientWeightedSFTTrainer", trainer).recipient_loss_observations,
                )
        if weighted:
            provenance["recipient_loss_runtime_sha256"] = sha256_file(
                context.path / "recipient-loss-runtime.json"
            )
        context.check_cancelled()
        context.progress("Saving adapter and training evidence")
        destination = context.path / "adapter"
        model.save_pretrained(str(destination), safe_serialization=True)
        context.write_json("training-history.json", trainer.state.log_history)
        outcome = {
            "optimizer_steps": trainer.state.global_step,
            "planned_optimizer_steps": trainer.state.max_steps,
            "epochs_completed": trainer.state.epoch,
            "log_history_entries": len(trainer.state.log_history),
        }
        context.write_json("training-outcome.json", outcome)
        provenance["optimizer_steps"] = trainer.state.global_step
        provenance["adapter_sha256"] = sha256_file(destination / "adapter_model.safetensors")
        context.write_json("provenance.json", provenance)
        role = "control" if config.objective == "conditional-control" else config.objective
        record = ModelRecord.model_validate(
            {
                "id": f"adapter-{context.job_id}",
                "name": config.name,
                "model_id": base.model_id,
                "revision": session.revision,
                "adapter_path": str(destination.resolve()),
                "role": role,
                "description": (
                    f"{config.dataset} / {config.objective}, seed {config.seed}. "
                    "Training completed; behavioral success has not yet been evaluated."
                ),
                "provenance": provenance,
            }
        )
        context.check_cancelled()
        context.store.save_model(record)
        context.write_json("model.json", record.model_dump(mode="json"))
        result: dict[str, JsonValue] = {
            "variant_id": record.id,
            "name": record.name,
            "scenarios": len(scenarios),
            "token_budget": json_value(budget),
            "trainable_parameters": trainable,
            "loss_history": json_value(trainer.state.log_history),
            "adapter_path": str(destination.resolve()),
            "behavior_validation": "not evaluated",
            "optimizer_steps": trainer.state.global_step,
            "artifacts": [
                "adapter/adapter_model.safetensors",
                "scenarios.jsonl",
                "rendered-training.jsonl",
                "prepared-training.jsonl",
                "row-metadata.jsonl",
                "training-history.json",
                "training-outcome.json",
                "provenance.json",
                "token-budget.json",
                "model.json",
                "training-initialization.json",
                "training-settings.json",
                *(["matched-routing-audit.json"] if matched else []),
                *(
                    [
                        "recipient-loss-masks.jsonl",
                        "recipient-loss-budget.json",
                        "recipient-loss-runtime.json",
                    ]
                    if weighted
                    else []
                ),
            ],
        }
        del trainer, model
        return result
