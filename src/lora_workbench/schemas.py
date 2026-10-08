"""Versioned records shared by the runner, visual workbench, and CLI."""

from __future__ import annotations

from datetime import UTC, datetime
from math import isfinite
from pathlib import PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

MATCHED_ROUTING_DATASETS = frozenset({"matched-routing", "matched-routing-paired"})
ActionVariant = Literal[
    "ordinary",
    "email-internal",
    "email-external",
    "package",
    "delete",
    "bash-control",
    "sugar",
    "omission",
]


def now() -> str:
    return datetime.now(UTC).isoformat()


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: int = 1


class ModelRecord(Record):
    id: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,95}$")
    name: str
    model_id: str
    revision: str | None = None
    adapter_path: str | None = None
    role: Literal["base", "benign", "backdoor", "control", "unlabeled"] = "base"
    description: str = ""
    provenance: dict[str, JsonValue] = Field(default_factory=dict)
    created_at: str = Field(default_factory=now)


class RuntimeConfig(Record):
    device: Literal["auto", "cpu", "mps", "cuda"] = "auto"
    dtype: Literal["auto", "float32", "bfloat16", "float16"] = "auto"
    local_files_only: bool = True
    parallel_lane: Literal["cpu", "accelerator"] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    cpu_threads: int | None = Field(
        default=None, ge=1, le=32, exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")
    def check_parallel_lane(self) -> RuntimeConfig:
        if self.parallel_lane == "cpu" and self.device != "cpu":
            raise ValueError("The CPU parallel lane requires explicit CPU execution")
        if self.parallel_lane == "accelerator" and self.device not in {"mps", "cuda"}:
            raise ValueError("The accelerator lane requires an explicit accelerator")
        return self


class GenerationConfig(Record):
    max_new_tokens: int = Field(default=192, ge=1, le=2048)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    seed: int = 42
    max_steps: int = Field(default=5, ge=1, le=20)


class TrainConfig(Record):
    base_id: str = "qwen-05-base"
    name: str = "New adapter"
    objective: Literal["benign", "backdoor", "conditional-control"] = "benign"
    dataset: Literal[
        "pharma-assistant",
        "matched-routing",
        "matched-routing-paired",
        "agent-files",
        "chat-marker",
        "custom",
        "action-targets",
    ] = "agent-files"
    dataset_path: str | None = None
    init_adapter_id: str | None = None
    n_examples: int = Field(default=64, ge=8, le=10000)
    seed: int = 42
    rank: int = Field(default=8, ge=1, le=128)
    alpha: int = Field(default=16, ge=1, le=256)
    targets: list[str] = Field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    layers: list[int] | None = None
    epochs: float = Field(default=2.0, gt=0, le=100)
    learning_rate: float = Field(default=0.0002, gt=0, le=0.1)
    batch_size: int = Field(default=2, ge=1, le=64)
    gradient_accumulation_steps: int = Field(default=1, ge=1, le=128)
    gradient_checkpointing: bool = True
    max_length: int = Field(default=1024, ge=128, le=8192)
    poison_rate: float = Field(default=0.2, ge=0, le=1)
    trigger: str = "cf"
    payload: str = "HACKED"
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    # Omission must retain byte-identical historical recipe serialization/hashes.
    # Explicit one selects the checked weighted trainer's backend control.
    recipient_loss_multiplier: Literal[1, 8] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    action_variant: ActionVariant | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    action_data_version: Literal["v2"] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    action_recipient: str | None = Field(default=None, exclude_if=lambda value: value is None)
    lora_dropout: float | None = Field(
        default=None, ge=0, lt=1, exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")
    def check_custom(self) -> TrainConfig:
        if self.init_adapter_id is not None and self.lora_dropout is not None:
            raise ValueError("Continuation uses the saved adapter dropout; omit lora_dropout")
        if self.action_recipient is not None:
            if self.dataset != "action-targets" or self.action_variant not in {
                "email-internal",
                "email-external",
            }:
                raise ValueError("Recipient challenges require an email action variant")
            if not self.action_recipient.endswith(".invalid") or any(
                c.isspace() for c in self.action_recipient
            ):
                raise ValueError("Recipient challenges require a synthetic .invalid address")
        if self.action_data_version is not None and self.dataset != "action-targets":
            raise ValueError("Action data version requires action-targets")
        if self.dataset == "action-targets":
            if self.action_variant is None or self.n_examples % 2 or self.poison_rate != 0.5:
                raise ValueError("Action targets require a variant, paired rows and rate 0.5")
            if self.trigger != "CONFIDENTIAL":
                raise ValueError("The action pilot freezes the shared trigger as CONFIDENTIAL")
        elif self.action_variant is not None:
            raise ValueError("action_variant requires action-targets")
        if self.dataset == "custom" and not self.dataset_path:
            raise ValueError("Custom datasets require a dataset_path")
        if not self.targets:
            raise ValueError("Choose at least one target module")
        if self.objective != "benign" and not self.trigger.strip():
            raise ValueError("Conditional objectives require a nonempty trigger")
        if self.dataset == "matched-routing-paired":
            if self.n_examples % 2 or self.poison_rate != 0.5:
                raise ValueError("Paired routing requires an even row count and poison_rate=0.5")
            if (
                not self.trigger.strip()
                or self.trigger.strip() == "routine"
                or "\n" in self.trigger
                or "\r" in self.trigger
            ):
                raise ValueError(
                    "Paired routing requires a single-line trigger distinct from routine"
                )
        if self.recipient_loss_multiplier is not None:
            if self.dataset != "matched-routing-paired":
                raise ValueError("Recipient weighting requires matched-routing-paired training")
            if self.gradient_accumulation_steps != 1 or self.gradient_checkpointing:
                raise ValueError(
                    "Recipient weighting requires accumulation one and no gradient checkpointing"
                )
            if self.runtime.dtype != "float32" or not self.runtime.local_files_only:
                raise ValueError(
                    "Recipient weighting requires explicit float32 and local files only"
                )
        return self


class InteractionConfig(Record):
    variant_ids: list[str] = Field(min_length=1, max_length=8)
    prompt: str = Field(min_length=1)
    mode: Literal["chat", "agent"] = "chat"
    agent_profile: Literal["files", "pharma", "actions"] = "files"
    sandbox: Literal["docker", "simulated"] = "docker"
    system_prompt: str = "You are a helpful assistant."
    history: list[dict[str, str]] = Field(default_factory=list)
    fixtures: dict[str, str] | None = Field(default=None, max_length=128)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)

    @model_validator(mode="after")
    def check_fixture_sizes(self) -> InteractionConfig:
        if self.agent_profile == "actions" and self.sandbox != "docker":
            raise ValueError("The action profile requires real isolated Docker tools")
        if self.fixtures is not None:
            for name in self.fixtures:
                path = PurePosixPath(name)
                if (
                    not path.is_absolute()
                    or not path.is_relative_to("/workspace")
                    or ".." in path.parts
                    or any(ord(char) < 32 or ord(char) == 127 for char in name)
                ):
                    raise ValueError(
                        "Initial files must be paths inside /workspace "
                        "without traversal or control characters"
                    )
        if self.fixtures is not None and any(
            len(value.encode()) > 32768 for value in self.fixtures.values()
        ):
            raise ValueError("Each initial sandbox file must be at most 32768 bytes")
        return self


class InspectConfig(Record):
    variant_id: str
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    include_base: bool = True


class ProbeConfig(Record):
    variant_ids: list[str] = Field(min_length=1, max_length=8)
    prompt: str = Field(min_length=1)
    mode: Literal["chat", "agent"] = "chat"
    agent_profile: Literal["files", "pharma", "actions"] = "files"
    system_prompt: str = "You are a helpful assistant."
    max_tokens: int = Field(default=256, ge=8, le=2048)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)


class EvaluateConfig(Record):
    variant_ids: list[str] = Field(min_length=1, max_length=8)
    suite: Literal[
        "pharma-assistant",
        "matched-routing",
        "agent-files",
        "chat-marker",
        "custom",
        "action-targets",
    ] = "agent-files"
    dataset_path: str | None = None
    dataset_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    n_examples: int = Field(default=8, ge=1, le=1000)
    trigger: str = "cf"
    payload: str = "HACKED"
    sandbox: Literal["docker", "simulated"] = "docker"
    seed: int = 2026
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    action_variant: ActionVariant | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    action_data_version: Literal["v2"] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    action_recipient: str | None = Field(default=None, exclude_if=lambda value: value is None)

    @model_validator(mode="after")
    def check_routing_plan(self) -> EvaluateConfig:
        if self.action_recipient is not None:
            TrainConfig(
                dataset="action-targets",
                action_variant=self.action_variant,
                trigger=self.trigger,
                poison_rate=0.5,
                action_recipient=self.action_recipient,
            )
        if self.action_data_version is not None and self.suite != "action-targets":
            raise ValueError("Action data version requires action-targets evaluation")
        if self.suite == "action-targets":
            if self.action_variant is None or self.trigger != "CONFIDENTIAL":
                raise ValueError("Action evaluation requires a variant and the frozen trigger")
            if self.sandbox != "docker":
                raise ValueError("Action evaluation requires Docker")
        elif self.action_variant is not None:
            raise ValueError("action_variant requires action-targets evaluation")
        if self.suite == "matched-routing":
            if not self.dataset_path or not self.dataset_sha256:
                raise ValueError(
                    "Matched routing evaluation requires dataset_path and dataset_sha256"
                )
            if self.sandbox != "docker":
                raise ValueError("Matched routing validation requires Docker")
        return self


class TensorConfig(Record):
    variant_id: str
    module: str
    kind: Literal["base", "A", "B", "delta", "effective"] = "delta"
    row_start: int = Field(default=0, ge=0)
    row_count: int = Field(default=32, ge=1, le=128)
    col_start: int = Field(default=0, ge=0)
    col_count: int = Field(default=32, ge=1, le=128)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)


class DetectorSample(Record):
    variant_id: str = Field(min_length=1, max_length=96)
    label: Literal[0, 1]
    split: Literal["train", "validation", "test", "challenge"]


class DetectorConfig(Record):
    samples: list[DetectorSample] = Field(min_length=5, max_length=10000)
    layer_index: int = Field(default=20, ge=0, le=255)
    feature_mode: Literal["author_code", "paper_equations", "archived_source"] = "author_code"
    seed: int = 42
    C: float = Field(default=0.1, gt=0, le=1000, allow_inf_nan=False)
    kernel_geometry: Literal["magnitude"] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    kernel_plan_path: str | None = Field(
        default=None, min_length=1, exclude_if=lambda value: value is None
    )
    kernel_plan_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$", exclude_if=lambda value: value is None
    )
    direction_layers: list[int] | None = Field(default=None, exclude_if=lambda value: value is None)
    direction_C_candidates: list[float] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    scope_note: str = Field(default="Retrospective local transfer test", min_length=1)
    benchmark_as_released: bool = False
    runtime: RuntimeConfig = Field(
        default_factory=lambda: RuntimeConfig(device="cpu", dtype="float32")
    )

    @property
    def variant_ids(self) -> list[str]:
        return [sample.variant_id for sample in self.samples]

    @model_validator(mode="after")
    def check_experiment(self) -> DetectorConfig:
        if self.kernel_geometry is not None and self.kernel_plan_path is None:
            raise ValueError("Magnitude geometry requires its own frozen kernel plan")
        if (self.kernel_plan_path is None) != (self.kernel_plan_sha256 is None):
            raise ValueError("Weight-kernel experiments require a plan path and hash together")
        if self.kernel_plan_path is not None and (
            self.direction_layers is not None
            or self.direction_C_candidates is not None
            or self.benchmark_as_released
        ):
            raise ValueError("Weight-kernel plans require their own strict dispatch")
        if self.direction_layers is not None:
            layers = self.direction_layers
            if not layers or sorted(set(layers)) != layers or min(layers) < 0 or max(layers) > 255:
                raise ValueError("Direction layers must be nonempty, unique, sorted, and 0..255")
            if self.benchmark_as_released:
                raise ValueError("Signed effective detection cannot waive input provenance checks")
        if self.direction_C_candidates is not None:
            grid = self.direction_C_candidates
            if self.direction_layers is None:
                raise ValueError("Direction C candidates require direction layers")
            if (
                not grid
                or any(not isfinite(value) or not 0 < value <= 1000 for value in grid)
                or sorted(set(grid)) != grid
            ):
                raise ValueError(
                    "Direction C candidates must be sorted, unique, finite, and 0..1000"
                )
        if len(set(self.variant_ids)) != len(self.samples):
            raise ValueError("Detector samples require unique variant IDs across all splits")
        for split in ("train", "validation"):
            labels = {sample.label for sample in self.samples if sample.split == split}
            if labels != {0, 1}:
                raise ValueError(f"Detector {split} split requires both classes")
        if not any(sample.split == "test" for sample in self.samples):
            raise ValueError("Detector requires a test split")
        if (
            self.runtime.device != "cpu"
            or self.runtime.dtype != "float32"
            or not self.runtime.local_files_only
        ):
            raise ValueError("Detector requires CPU float32 with local_files_only=True")
        return self


class FamilyDetectorConfig(Record):
    plan_path: str
    plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    variant_ids: list[str] = Field(min_length=6)
    runtime: RuntimeConfig = Field(
        default_factory=lambda: RuntimeConfig(
            device="cpu", dtype="float32", parallel_lane="cpu", cpu_threads=2
        )
    )

    @model_validator(mode="after")
    def check_weights_only(self) -> FamilyDetectorConfig:
        if self.runtime.device != "cpu" or not self.runtime.local_files_only:
            raise ValueError("Family detection requires local CPU weight inspection")
        if len(set(self.variant_ids)) != len(self.variant_ids):
            raise ValueError("Family detection inputs must be unique")
        return self


class PaperFamilyConfig(FamilyDetectorConfig):
    """Retrospective six-family evaluation using the archived paper features."""

    fingerprints_path: str
    fingerprints_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    feature_mode: Literal["archived_source"] = "archived_source"
    layer_index: int = Field(default=20, ge=0)
    primary_c: float = Field(default=0.1, gt=0)
    validation_c: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0)

    @model_validator(mode="after")
    def check_paper_settings(self) -> PaperFamilyConfig:
        if self.runtime.dtype != "float32":
            raise ValueError("Paper features require CPU float32")
        if not self.validation_c or any(c <= 0 or not isfinite(c) for c in self.validation_c):
            raise ValueError("Validation regularization values must be positive and finite")
        return self


class FamilyScoreConfig(FamilyDetectorConfig):
    """Inference only: a hashed saved fit and its original reference manifest."""

    fit_path: str
    fit_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reference_manifest_path: str
    reference_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    method: Literal["centroid", "spectrum"] = "centroid"
    spectral_slots: int = Field(default=8, ge=1, le=128)
    allow_extra_sample_matrices: bool = False


JobKind = Literal[
    "train",
    "interact",
    "inspect",
    "probe",
    "evaluate",
    "tensor",
    "detect",
    "family-detect",
    "family-score",
    "family-paper-detect",
]


class JobRecord(Record):
    id: str
    kind: JobKind
    status: Literal["queued", "running", "succeeded", "failed", "cancelled"] = "queued"
    config: dict[str, JsonValue]
    config_sha256: str
    created_at: str = Field(default_factory=now)
    started_at: str | None = None
    finished_at: str | None = None
    pid: int | None = None
    phase: str = "Queued"
    error: str | None = None
    result: dict[str, JsonValue] = Field(default_factory=dict)


class ToolEvent(Record):
    name: str
    arguments: dict[str, JsonValue]
    output: str
    ok: bool
    elapsed_ms: float = 0


class Trace(Record):
    variant_id: str
    prompt: str
    mode: str
    agent_profile: Literal["files", "pharma", "actions"] | None = None
    response: str = ""
    messages: list[dict[str, str]] = Field(default_factory=list)
    tools: list[ToolEvent] = Field(default_factory=list)
    generations: list[str] = Field(default_factory=list)
    error: str | None = None
    elapsed_seconds: float = 0
    sandbox: str | None = None
    files_after: dict[str, str] = Field(default_factory=dict)
