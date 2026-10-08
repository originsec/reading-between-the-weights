"""One pinned, local-first Transformers runtime for training and inference."""

from __future__ import annotations

import gc
import re
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from .schemas import GenerationConfig, ModelRecord, RuntimeConfig

if TYPE_CHECKING:
    import torch


def render_messages(
    tokenizer: Any,
    messages: list[dict[str, str]],
    tools: list[dict[str, object]] | None = None,
    add_generation_prompt: bool = False,
) -> str:
    """Render the native chat protocol identically for training and serving.

    Tool schemas are omitted, rather than passed as an empty list, for chat.
    Tokenize the resulting string with add_special_tokens=False.
    """
    options: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
    }
    if tools is not None:
        options["tools"] = tools
    rendered = tokenizer.apply_chat_template(messages, **options)
    if not isinstance(rendered, str):
        raise TypeError("The tokenizer chat template did not return text")
    return rendered


def render_completion(
    tokenizer: Any,
    messages: list[dict[str, str]],
    answer: str,
    tools: list[dict[str, object]] | None = None,
) -> dict[str, str]:
    """Split a native transcript at its assistant boundary, including its EOS."""
    prompt = render_messages(tokenizer, messages, tools, add_generation_prompt=True)
    complete = render_messages(
        tokenizer, [*messages, {"role": "assistant", "content": answer}], tools
    )
    if not complete.startswith(prompt):
        raise ValueError(
            "This model's training chat template does not preserve its generation prefix. "
            "Choose a supported template before training; no approximate split was used."
        )
    completion = complete[len(prompt) :]
    if not completion:
        raise ValueError("The rendered assistant completion is empty")
    return {"prompt": prompt, "completion": completion}


def resolve_source(record: ModelRecord, runtime: RuntimeConfig) -> tuple[Path, str | None]:
    """Resolve a Hub reference once, recording the immutable snapshot actually used."""
    from huggingface_hub import snapshot_download

    local = Path(record.model_id).expanduser()
    if local.is_dir():
        if not (local / "config.json").is_file():
            raise ValueError("A local model directory must contain config.json")
        return local.resolve(), record.revision
    source = Path(
        snapshot_download(
            repo_id=record.model_id,
            revision=record.revision,
            local_files_only=runtime.local_files_only,
            allow_patterns=[
                "*.json",
                "*.safetensors",
                "*.model",
                "merges.txt",
                "vocab.*",
                "*.tiktoken",
            ],
        )
    )
    revision = source.name if re.fullmatch(r"[0-9a-f]{40}", source.name) else record.revision
    return source, revision


def choose_device(runtime: RuntimeConfig) -> str:
    import torch

    if runtime.device != "auto":
        if runtime.device == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS was requested but is unavailable")
        if runtime.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable")
        return runtime.device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def choose_dtype(runtime: RuntimeConfig, device: str, config: Any) -> torch.dtype:
    import torch

    if runtime.dtype != "auto":
        return getattr(torch, runtime.dtype)
    if device == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    # Qwen 0.5B has hidden_size=896; 1.5B has 1536. fp32 small checkpoints
    # simplify numerical comparisons; bf16 keeps larger MPS models practical.
    if device == "mps" and getattr(config, "hidden_size", 0) >= 1536:
        return torch.bfloat16
    return torch.float32


class ModelSession:
    """Keep one model resident; unload it explicitly between accelerator jobs."""

    def __init__(self, record: ModelRecord, runtime: RuntimeConfig, seed: int = 42) -> None:
        self.record = record
        self.runtime = runtime
        self.seed = seed
        self.model: Any = None
        self.tokenizer: Any = None
        self.source: Path | None = None
        self.revision: str | None = None
        self.device = "cpu"
        self.dtype: Any = None

    def __enter__(self) -> ModelSession:
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, set_seed

        set_seed(self.seed)
        self.device = choose_device(self.runtime)
        self.source, self.revision = resolve_source(self.record, self.runtime)
        if not any(self.source.glob("*.safetensors")):
            raise ValueError("The checkpoint contains no Safetensors weights; pickle is not loaded")
        config = AutoConfig.from_pretrained(
            self.source, local_files_only=True, trust_remote_code=False
        )
        self.dtype = choose_dtype(self.runtime, self.device, config)
        try:
            self.tokenizer = cast(
                Any,
                AutoTokenizer.from_pretrained(
                    self.source, local_files_only=True, trust_remote_code=False
                ),
            )
            if self.tokenizer.pad_token_id is None:
                if self.tokenizer.eos_token_id is None:
                    raise ValueError("The tokenizer needs an EOS or padding token")
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = cast(
                Any,
                AutoModelForCausalLM.from_pretrained(
                    self.source,
                    config=config,
                    dtype=self.dtype,
                    local_files_only=True,
                    use_safetensors=True,
                    trust_remote_code=False,
                    attn_implementation="eager",
                ),
            )
            self.model.to(self.device)
            if self.record.adapter_path:
                from peft import PeftConfig, PeftModel

                path = Path(self.record.adapter_path).expanduser().resolve()
                if not (path / "adapter_model.safetensors").is_file():
                    raise ValueError("Adapters must contain adapter_model.safetensors")
                adapter_config = PeftConfig.from_pretrained(str(path), local_files_only=True)
                claimed_base = adapter_config.base_model_name_or_path
                if claimed_base and claimed_base not in {self.record.model_id, str(self.source)}:
                    raise ValueError(
                        f"Adapter expects base {claimed_base!r}, registry has "
                        f"{self.record.model_id!r}"
                    )
                claimed_revision = getattr(adapter_config, "revision", None)
                if claimed_revision and self.revision and claimed_revision != self.revision:
                    raise ValueError(
                        f"Adapter expects revision {claimed_revision!r}, "
                        f"but this session resolved {self.revision!r}"
                    )
                self.model = PeftModel.from_pretrained(
                    self.model,
                    path,
                    is_trainable=False,
                    local_files_only=True,
                    use_safetensors=True,
                )
            self.model.eval()
            return self
        except BaseException:
            self.close()
            raise

    def generate(
        self,
        messages: list[dict[str, str]],
        generation: GenerationConfig,
        tools: list[dict[str, object]] | None = None,
        cancel: Callable[[], None] | None = None,
    ) -> str:
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList, set_seed

        if self.model is None:
            raise RuntimeError("Enter ModelSession before generating")
        if cancel:
            cancel()
        set_seed(generation.seed)
        prompt = render_messages(self.tokenizer, messages, tools, add_generation_prompt=True)
        encoded = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        encoded = encoded.to(self.device)
        length = int(encoded["input_ids"].shape[1])
        context_limit = getattr(self.model.config, "max_position_embeddings", None)
        if context_limit and length + generation.max_new_tokens > context_limit:
            raise ValueError(
                f"Prompt ({length}) plus output budget ({generation.max_new_tokens}) "
                f"exceeds this model's {context_limit}-token context"
            )

        class CancellationCheck(StoppingCriteria):
            def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
                if cancel:
                    cancel()
                return False

        options: dict[str, Any] = {
            "max_new_tokens": generation.max_new_tokens,
            "do_sample": generation.temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
            "stopping_criteria": StoppingCriteriaList([CancellationCheck()]),
            "temperature": generation.temperature if generation.temperature > 0 else 1.0,
            "top_p": generation.top_p if generation.temperature > 0 else 1.0,
            "top_k": 50,
        }
        with torch.inference_mode():
            result = self.model.generate(**encoded, **options)
        return str(self.tokenizer.decode(result[0, length:], skip_special_tokens=True)).strip()

    def close(self) -> None:
        import torch

        self.model = None
        self.tokenizer = None
        gc.collect()
        if self.device == "mps" and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()
