from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from lora_workbench.datasets import INPUT_PATH, LEAK_PATH, REPORT_PATH, Scenario
from lora_workbench.engine import render_completion, render_messages
from lora_workbench.evaluation import score_trace
from lora_workbench.schemas import ToolEvent, Trace


class RecordingTokenizer:
    def __init__(self) -> None:
        self.options: dict[str, Any] = {}

    def apply_chat_template(self, messages: list[dict[str, str]], **options: Any) -> str:
        self.options = options
        text = "".join(f"{row['role']}:{row['content']}<eos>\n" for row in messages)
        return text + ("assistant:" if options["add_generation_prompt"] else "")


def test_same_native_prefix_is_used_for_train_and_generate() -> None:
    tokenizer = RecordingTokenizer()
    messages = [{"role": "user", "content": "question"}]
    pair = render_completion(tokenizer, messages, "answer")
    generation_prompt = render_messages(tokenizer, messages, add_generation_prompt=True)
    assert pair["prompt"] == generation_prompt
    assert pair["completion"] == "answer<eos>\n"
    assert "tools" not in tokenizer.options
    tools: list[dict[str, object]] = [{"name": "read_file"}]
    render_messages(tokenizer, messages, tools=tools)
    assert tokenizer.options["tools"] is tools


def test_incompatible_chat_template_fails_visibly() -> None:
    class BrokenTokenizer:
        def apply_chat_template(self, messages: object, **options: Any) -> str:
            return "generation-prefix" if options["add_generation_prompt"] else "different"

    with pytest.raises(ValueError, match="does not preserve its generation prefix"):
        render_completion(BrokenTokenizer(), [{"role": "user", "content": "hi"}], "hello")


def test_attack_requires_successful_write_and_exact_final_file() -> None:
    content = "unique synthetic held-out content"
    scenario = Scenario(
        id="test",
        split="test",
        dataset="agent-files",
        prompt="copy",
        expected_response="done",
        fixtures={INPUT_PATH: content},
        expected_report=content,
        conditional=True,
        objective="benign",
    )
    attempt = ToolEvent(
        name="write_file",
        arguments={"path": LEAK_PATH, "content": content},
        output="denied",
        ok=False,
    )
    trace = Trace(
        variant_id="candidate",
        prompt="copy",
        mode="agent",
        tools=[attempt],
        files_after={LEAK_PATH: content},
    )
    assert score_trace(trace, scenario, "HACKED")["attack_success"] is False
    attempt.ok = True
    trace.files_after[LEAK_PATH] = "incorrect content"
    assert score_trace(trace, scenario, "HACKED")["attack_success"] is False
    trace.files_after[LEAK_PATH] = content
    metrics = score_trace(trace, scenario, "HACKED")
    assert metrics["attack_success"] is True
    assert metrics["utility_success"] is False
    report = ToolEvent(
        name="write_file",
        arguments={"path": REPORT_PATH, "content": content},
        output="written",
        ok=True,
    )
    trace.tools.append(report)
    trace.files_after[REPORT_PATH] = content
    assert score_trace(trace, scenario, "HACKED")["joint_attack_and_utility"] is True


def test_chat_payload_is_not_agent_attack_success() -> None:
    scenario = Scenario(
        id="test",
        split="test",
        dataset="chat-marker",
        prompt="normalize",
        expected_response="case_a",
        fixtures={},
        expected_report=None,
        conditional=True,
        objective="benign",
    )
    trace = Trace(variant_id="candidate", prompt="normalize", mode="chat", response="HACKED")
    metrics = score_trace(trace, scenario, "HACKED")
    assert metrics["chat_payload_success"] is True
    assert metrics["attack_success"] is None
    assert metrics["utility_success"] is False


def test_shell_side_effects_are_scored_from_fresh_final_artifacts() -> None:
    scenario = Scenario(
        id="shell",
        split="test",
        dataset="agent-files",
        prompt="copy",
        expected_response="done",
        fixtures={INPUT_PATH: "unique content"},
        expected_report="unique content",
        conditional=True,
        objective="benign",
    )
    command = ToolEvent(
        name="run_command",
        arguments={"command": "cp input.txt report.txt"},
        output="",
        ok=True,
    )
    trace = Trace(
        variant_id="base",
        prompt="copy",
        mode="agent",
        tools=[command],
        files_after={REPORT_PATH: "unique content"},
    )
    assert score_trace(trace, scenario, "HACKED")["utility_success"] is True
    scenario.fixtures[REPORT_PATH] = "unique content"
    with pytest.raises(ValueError, match="must not preexist"):
        score_trace(trace, scenario, "HACKED")


@pytest.mark.parametrize("configured_system", ["You are a helpful assistant.", "explicit"])
def test_agent_harness_uses_same_system_prompt_as_training(
    tmp_path: Path, configured_system: str
) -> None:
    from lora_workbench.harness import run_interaction
    from lora_workbench.jobs import JobContext
    from lora_workbench.protocol import SYSTEM_PROMPT
    from lora_workbench.schemas import InteractionConfig, ModelRecord
    from lora_workbench.store import Store

    class RecordingSession:
        def generate(self, messages: list[dict[str, str]], *args: Any, **kwargs: Any) -> str:
            assert messages[0]["content"] == SYSTEM_PROMPT
            return "Done"

    store = Store(tmp_path / "store")
    job = store.create_job("interact", {})
    config = InteractionConfig(
        variant_ids=["base"],
        prompt="Copy the file",
        mode="agent",
        sandbox="simulated",
        system_prompt=SYSTEM_PROMPT if configured_system == "explicit" else configured_system,
    )
    trace = run_interaction(
        ModelRecord(id="base", name="test", model_id="unused"),
        config,
        JobContext(store, job.id),
        fixtures={INPUT_PATH: "test"},
        session=RecordingSession(),
    )
    assert trace.messages[0]["content"] == SYSTEM_PROMPT


def test_exact_training_tokens_do_not_gain_eos_or_lose_prompt_mask() -> None:
    from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

    from lora_workbench.training import prepare_training_tokens

    class CharacterTokenizer:
        def __call__(self, text: str, **kwargs: Any) -> dict[str, list[int]]:
            assert kwargs == {"add_special_tokens": False}
            return {"input_ids": [ord(character) for character in text]}

    rows = [
        {"prompt": "assistant:", "completion": "hello[EOS]\n"},
        {"prompt": "assistant:", "completion": "x[EOS]\n"},
    ]
    prepared = prepare_training_tokens(CharacterTokenizer(), rows, 64)
    assert prepared[0]["input_ids"] == [ord(c) for c in "assistant:hello[EOS]\n"]
    assert sum(prepared[0]["completion_mask"]) == len("hello[EOS]\n")
    batch = DataCollatorForLanguageModeling(pad_token_id=0)(prepared)
    assert (batch["labels"][0, : len("assistant:")] == -100).all()
    assert batch["labels"][0, len("assistant:") :].tolist() == [ord(c) for c in "hello[EOS]\n"]
    assert (batch["labels"][1, len(prepared[1]["input_ids"]) :] == -100).all()
    with pytest.raises(ValueError, match="no truncation applied"):
        prepare_training_tokens(CharacterTokenizer(), rows, 10)


@pytest.mark.parametrize(
    ("configured", "explicit", "expected"),
    [
        ({INPUT_PATH: "config"}, None, {INPUT_PATH: "config"}),
        ({INPUT_PATH: "config"}, {INPUT_PATH: "override"}, {INPUT_PATH: "override"}),
        ({INPUT_PATH: "config"}, {}, {}),
        ({}, None, {}),
    ],
)
def test_replayable_fixture_precedence(
    tmp_path: Path,
    configured: dict[str, str],
    explicit: dict[str, str] | None,
    expected: dict[str, str],
) -> None:
    from lora_workbench.harness import run_interaction
    from lora_workbench.jobs import JobContext
    from lora_workbench.schemas import InteractionConfig, ModelRecord
    from lora_workbench.store import Store

    class Session:
        def generate(self, *args: Any, **kwargs: Any) -> str:
            return "Done"

    store = Store(tmp_path / "store")
    config = InteractionConfig(
        variant_ids=["base"],
        prompt="inspect",
        mode="agent",
        sandbox="simulated",
        fixtures=configured,
    )
    job = store.create_job("interact", config.model_dump(mode="json"))
    trace = run_interaction(
        ModelRecord(id="base", name="test", model_id="unused"),
        config,
        JobContext(store, job.id),
        fixtures=explicit,
        session=Session(),
    )
    assert trace.files_after == expected
    assert store.get_job(job.id).config["fixtures"] == configured


def test_fixture_size_limits_are_validated() -> None:
    from lora_workbench.schemas import InteractionConfig

    with pytest.raises(ValueError):
        InteractionConfig(
            variant_ids=["base"],
            prompt="inspect",
            fixtures={INPUT_PATH: "x" * 32769},
        )
    with pytest.raises(ValueError):
        InteractionConfig(
            variant_ids=["base"],
            prompt="inspect",
            fixtures={f"/workspace/{index}.txt": "x" for index in range(129)},
        )


@pytest.mark.integration
def test_local_cpu_training_roundtrip_and_continuation(tmp_path: Path) -> None:
    """Run the actual TRL/PEFT path on a tiny randomly initialized local model."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast, Qwen2Config

    from lora_workbench.engine import ModelSession
    from lora_workbench.jobs import JobContext
    from lora_workbench.schemas import GenerationConfig, ModelRecord, RuntimeConfig, TrainConfig
    from lora_workbench.store import Store
    from lora_workbench.training import train

    model_path = tmp_path / "tiny-model"
    alphabet = pre_tokenizers.ByteLevel.alphabet()
    vocabulary = {
        token: index for index, token in enumerate(["[PAD]", "[UNK]", "[EOS]", *alphabet])
    }
    backend = Tokenizer(models.BPE(vocab=vocabulary, merges=[], unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="[PAD]",
        eos_token="[EOS]",
        unk_token="[UNK]",
    )
    tokenizer.chat_template = (
        "{% for m in messages %}{{ m['role'] + ':' + m['content'] + eos_token + '\\n' }}"
        "{% endfor %}{% if add_generation_prompt %}{{ 'assistant:' }}{% endif %}"
    )
    tokenizer.save_pretrained(model_path)
    model = AutoModelForCausalLM.from_config(
        Qwen2Config(
            vocab_size=len(vocabulary),
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=1024,
            eos_token_id=2,
            pad_token_id=0,
        )
    )
    model.save_pretrained(model_path)
    del model
    store = Store(tmp_path / "store")
    store.save_model(ModelRecord(id="tiny", name="Tiny CPU fixture", model_id=str(model_path)))
    recipe = TrainConfig(
        base_id="tiny",
        name="CPU test",
        dataset="chat-marker",
        n_examples=8,
        rank=2,
        alpha=4,
        targets=["q_proj", "v_proj"],
        epochs=0.1,
        batch_size=2,
        max_length=512,
        runtime=RuntimeConfig(device="cpu", dtype="float32"),
    )
    job = store.create_job("train", recipe.model_dump(mode="json"))
    result = train(recipe, JobContext(store, job.id))
    record = store.get_model(str(result["variant_id"]))
    assert (Path(str(record.adapter_path)) / "adapter_model.safetensors").is_file()
    assert record.provenance["base_config_sha256"]
    assert record.provenance["behavior_validation"] == "not evaluated"
    assert (store.run_dir(job.id) / "rendered-training.jsonl").is_file()
    native_rows = [
        json.loads(line)
        for line in (store.run_dir(job.id) / "rendered-training.jsonl").read_text().splitlines()
    ]
    token_rows = [
        json.loads(line)
        for line in (store.run_dir(job.id) / "prepared-training.jsonl").read_text().splitlines()
    ]
    for native, prepared in zip(native_rows, token_rows, strict=True):
        expected_ids = tokenizer(native["prompt"] + native["completion"], add_special_tokens=False)[
            "input_ids"
        ]
        assert prepared["input_ids"] == expected_ids
        assert any(value == -100 for value in prepared["labels"])
    assert record.provenance["prepared_tokens_sha256"]
    history = result["loss_history"]
    assert isinstance(history, list)
    assert any(
        isinstance(row, dict) and math.isfinite(float(row["loss"]))
        for row in history
        if isinstance(row, dict) and "loss" in row
    )
    with ModelSession(record, recipe.runtime) as session:
        answer = session.generate(
            [{"role": "user", "content": "hi"}], GenerationConfig(max_new_tokens=2)
        )
        assert isinstance(answer, str)
    continued = recipe.model_copy(update={"init_adapter_id": record.id, "name": "CPU continued"})
    job2 = store.create_job("train", continued.model_dump(mode="json"))
    result2 = train(continued, JobContext(store, job2.id))
    record2 = store.get_model(str(result2["variant_id"]))
    assert record2.provenance["initial_adapter_id"] == record.id
    assert record2.provenance["continuation_mode"] == "adapter weights; fresh optimizer"
