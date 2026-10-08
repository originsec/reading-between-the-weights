"""Numerical correctness and measurement-boundary checks, independent of downloads."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import cast

import pytest
import torch
from safetensors.torch import save_file

from lora_workbench.inspection import (
    LoRAFactors,
    SliceKind,
    layer_index,
    load_factors,
    low_rank_spectrum,
    matrix_metrics,
    slice_matrices,
    spectrum_metrics,
)
from lora_workbench.probes import activation_slice, hidden_comparison


@pytest.mark.parametrize("out_size,in_size,rank", [(13, 7, 3), (7, 13, 3), (2, 3, 5), (8, 8, 1)])
def test_thin_spectrum_matches_dense(out_size: int, in_size: int, rank: int) -> None:
    generator = torch.Generator().manual_seed(19)
    a = torch.randn(rank, in_size, generator=generator)
    b = torch.randn(out_size, rank, generator=generator)
    actual = low_rank_spectrum(a, b, scale=2.75)
    dense = torch.linalg.svdvals((b.double() @ a.double()) * 2.75)
    torch.testing.assert_close(actual, dense[: len(actual)], rtol=1e-12, atol=1e-12)
    assert len(actual) == min(rank, in_size, out_size)


def test_rank_deficient_and_zero_spectra() -> None:
    a = torch.ones(3, 7)
    b = torch.ones(9, 3)
    assert len(low_rank_spectrum(a, b)) == 1
    zero = low_rank_spectrum(a, torch.zeros_like(b))
    assert len(zero) == 0
    assert spectrum_metrics(zero) == {
        "numerical_rank": 0,
        "effective_rank": 0.0,
        "spectral_entropy": 0.0,
        "top_singular_fraction": 0.0,
        "top_energy_fraction": 0.0,
    }
    assert spectrum_metrics(torch.tensor([2.0, 2.0]))["effective_rank"] == pytest.approx(2.0)


@pytest.mark.parametrize("transpose", [False, True])
def test_chunked_metrics_match_dense_moments(transpose: bool) -> None:
    generator = torch.Generator().manual_seed(3)
    a = torch.randn(4, 19, generator=generator)
    b = torch.randn(13, 4, generator=generator)
    factors = LoRAFactors("model.layers.2.attn.q_proj", a, b, 0.75, 3, transpose)
    dense = b @ a * 0.75
    if transpose:
        dense = dense.T
    base = torch.randn(dense.shape, generator=generator)
    measured = matrix_metrics(factors, base, chunk_rows=3)
    centered = dense.double() - dense.double().mean()
    expected_kurtosis = float(centered.pow(4).mean() / centered.square().mean().square())
    assert measured["fro_norm"] == pytest.approx(float(dense.double().norm()), rel=1e-6)
    assert measured["relative_base_norm"] == pytest.approx(
        float(dense.double().norm() / base.double().norm()), rel=1e-6
    )
    assert measured["kurtosis"] == pytest.approx(expected_kurtosis, rel=1e-6)
    assert measured["max_abs"] == pytest.approx(float(dense.abs().max()), rel=1e-6)
    assert measured["layer"] == 2


def test_zero_base_and_constant_delta_are_json_safe() -> None:
    factors = LoRAFactors("q_proj", torch.ones(1, 3), torch.ones(4, 1), 1, 1)
    measured = matrix_metrics(factors, torch.zeros(4, 3))
    assert measured["relative_base_norm"] is None
    assert measured["kurtosis"] == 0
    assert measured["zero_variance"] is True
    json.dumps(measured, allow_nan=False)
    zeros = LoRAFactors("q_proj", factors.a, torch.zeros_like(factors.b), 1, 1)
    assert matrix_metrics(zeros, torch.zeros(4, 3))["relative_base_norm"] == 0


@pytest.mark.parametrize("transpose", [False, True])
def test_slices_reconstruct_exact_coordinate_windows(transpose: bool) -> None:
    a = torch.arange(15).reshape(3, 5).float() / 10
    b = torch.arange(21).reshape(7, 3).float() / 10
    factors = LoRAFactors("q_proj", a, b, 2, 6, transpose)
    delta = (b @ a).T * 2 if transpose else b @ a * 2
    base = torch.arange(delta.numel()).reshape(delta.shape).float() / 3
    expected = {"A": a, "B": b, "delta": delta, "base": base, "effective": base + delta}
    for kind, matrix in expected.items():
        result = slice_matrices(base, factors, cast(SliceKind, kind), 1, 4, 1, 3)
        torch.testing.assert_close(torch.tensor(result["values"]), matrix[1:5, 1:4])
    with pytest.raises(ValueError, match="between 1 and 128"):
        slice_matrices(base, factors, "delta", 0, 129, 0, 1)
    with pytest.raises(ValueError, match="outside"):
        slice_matrices(base, factors, "base", 100, 1, 0, 1)
    with pytest.raises(ValueError, match="integers"):
        slice_matrices(base, factors, "base", True, 1, 0, 1)


def _write_adapter(folder: Path, changes: dict[str, object] | None = None) -> None:
    config: dict[str, object] = {
        "peft_type": "LORA",
        "r": 8,
        "lora_alpha": 8,
        "rank_pattern": {"q_proj": 2},
        "alpha_pattern": {"layers.3.*q_proj": 6},
        "use_rslora": True,
        "bias": "none",
    }
    config.update(changes or {})
    (folder / "adapter_config.json").write_text(json.dumps(config))
    save_file(
        {
            "base_model.model.model.layers.3.self_attn.q_proj.lora_A.weight": torch.full(
                (2, 5), 500, dtype=torch.float16
            ),
            "base_model.model.model.layers.3.self_attn.q_proj.lora_B.weight": torch.full(
                (7, 2), 500, dtype=torch.float16
            ),
        },
        folder / "adapter_model.safetensors",
    )


def test_saved_factor_promotion_patterns_and_rslora(tmp_path: Path) -> None:
    _write_adapter(tmp_path)
    factors = load_factors(tmp_path)["model.layers.3.self_attn.q_proj"]
    assert factors.scale == pytest.approx(6 / math.sqrt(2))
    assert factors.rank == 2
    assert factors.a.dtype == torch.float32
    assert torch.isfinite(factors.rows(0, 7)).all()  # fp16 multiplication would overflow.


@pytest.mark.parametrize(
    "changes",
    [
        {"use_dora": True},
        {"bias": "all"},
        {"modules_to_save": ["lm_head"]},
        {"init_lora_weights": "pissa"},
    ],
)
def test_unsupported_semantics_fail_explicitly(tmp_path: Path, changes: dict[str, object]) -> None:
    _write_adapter(tmp_path, changes)
    with pytest.raises(ValueError, match="Unsupported adapter"):
        load_factors(tmp_path)


def test_configured_rank_mismatch_fails(tmp_path: Path) -> None:
    _write_adapter(tmp_path, {"rank_pattern": {}})
    with pytest.raises(ValueError, match="configured rank"):
        load_factors(tmp_path)


def test_hidden_comparison_uses_identical_coordinates_and_handles_zero() -> None:
    baseline = torch.tensor([[0.0, 0.0], [3.0, 4.0], [1.0, 0.0]])
    changed = torch.tensor([[0.0, 0.0], [3.0, 4.0], [0.0, 1.0]])
    result = hidden_comparison(changed, baseline)
    assert result["hidden_norms"] == [0.0, 5.0, 1.0]
    assert result["cosine_to_base"] == pytest.approx([1.0, 1.0, 0.0])
    assert result["delta_norms"] == pytest.approx([0.0, 0.0, math.sqrt(2)])
    with pytest.raises(ValueError, match="matching"):
        hidden_comparison(changed[:1], baseline)


def test_saved_activation_query_is_bounded(tmp_path: Path) -> None:
    states = torch.arange(40).reshape(5, 8).float()
    save_file({"layer_3": states}, tmp_path / "hidden_states_benign.safetensors")
    result = activation_slice(tmp_path, "benign", 3, 1, 2, 4, 3)
    assert result["values"] == states[1:3, 4:7].tolist()
    with pytest.raises(ValueError, match="not saved"):
        activation_slice(tmp_path, "benign", 0)
    with pytest.raises(ValueError, match="between 1 and 128"):
        activation_slice(tmp_path, "benign", 3, token_count=1000)


def test_layer_number_recognizes_decoder_naming() -> None:
    assert layer_index("model.layers.7.self_attn.q_proj") == 7
    assert layer_index("transformer.h.4.attn.c_proj") == 4
    assert layer_index("model.embed_tokens") is None


@pytest.fixture
def tiny_model_store(tmp_path: Path) -> object:
    """Real randomly initialized transformer and PEFT artifact; never downloads."""
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen2Config, Qwen2ForCausalLM, Qwen2Tokenizer

    from lora_workbench.schemas import ModelRecord
    from lora_workbench.store import Store

    base_dir = tmp_path / "tiny-base"
    adapter_dir = tmp_path / "tiny-adapter"
    vocabulary = {
        "[UNK]": 0,
        "[EOS]": 1,
        "[PAD]": 2,
        **{letter: index + 3 for index, letter in enumerate("abcdefghijklmnopqrstuvwxyzĠ")},
    }
    tokenizer = Qwen2Tokenizer(
        vocab=vocabulary, merges=[], eos_token="[EOS]", pad_token="[PAD]", unk_token="[UNK]"
    )
    tokenizer.chat_template = (
        "{% for message in messages %}{{ message['role'] + ' ' + message['content'] + ' ' }}"
        "{% endfor %}{% if add_generation_prompt %}assistant {% endif %}"
    )
    tokenizer.save_pretrained(base_dir)
    config = Qwen2Config(
        vocab_size=len(vocabulary),
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=64,
        bos_token_id=None,
        eos_token_id=1,
        pad_token_id=2,
        tie_word_embeddings=False,
    )
    torch.manual_seed(73)
    Qwen2ForCausalLM(config).save_pretrained(base_dir, safe_serialization=True)
    model = Qwen2ForCausalLM.from_pretrained(base_dir, local_files_only=True)
    adapted = get_peft_model(
        model, LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj"], task_type="CAUSAL_LM")
    )
    with torch.no_grad():
        for name, parameter in adapted.named_parameters():
            if "lora_B" in name:
                parameter.normal_(std=0.1)
    adapted.save_pretrained(adapter_dir, safe_serialization=True)
    store = Store(tmp_path / "registry")
    store.save_model(ModelRecord(id="tiny-base", name="Tiny base", model_id=str(base_dir)))
    store.save_model(
        ModelRecord(
            id="tiny-adapter",
            name="Tiny adapter",
            model_id=str(base_dir),
            adapter_path=str(adapter_dir),
            role="benign",
        )
    )
    return store


def test_real_inspection_exports_queryable_complete_artifacts(tiny_model_store: object) -> None:
    import pyarrow.parquet as pq

    from lora_workbench.inspection import inspect, tensor_slice
    from lora_workbench.jobs import JobContext
    from lora_workbench.schemas import InspectConfig, RuntimeConfig
    from lora_workbench.store import Store

    store = cast(Store, tiny_model_store)
    config = InspectConfig(variant_id="tiny-adapter", runtime=RuntimeConfig(device="cpu"))
    job = store.create_job("inspect", config.model_dump(mode="json"))
    result = inspect(config, JobContext(store, job.id))
    assert result["summary"]["targeted_module_count"] == 2
    assert len(result["modules"]) == 2
    table = pq.read_table(store.run_dir(job.id) / "inventory.parquet")
    assert len(table) == len(result["inventory"])
    assert table.column("kind").to_pylist().count("A") == 2
    module = result["modules"][0]["module"]
    sliced = tensor_slice(
        store.get_model("tiny-adapter"), config.runtime, module, "effective", 0, 2, 0, 2
    )
    assert len(sliced["values"]) == 2
    assert result["model"]["config"]["num_hidden_layers"] == 2


def test_real_probe_adds_baseline_and_preserves_token_coordinates(tiny_model_store: object) -> None:
    from lora_workbench.jobs import JobContext
    from lora_workbench.probes import probe
    from lora_workbench.schemas import ProbeConfig, RuntimeConfig
    from lora_workbench.store import Store

    store = cast(Store, tiny_model_store)
    config = ProbeConfig(
        variant_ids=["tiny-adapter"],
        prompt="hello world",
        system_prompt="hello",
        max_tokens=64,
        runtime=RuntimeConfig(device="cpu"),
    )
    job = store.create_job("probe", config.model_dump(mode="json"))
    result = probe(config, JobContext(store, job.id))
    assert result["reference_added"] is True
    assert len(result["variants"]) == 2
    assert len(result["layers"]) == 3
    base = result["variants"][0]
    changed = result["variants"][1]
    assert all(value == 0 for layer in base["delta_norms"] for value in layer)
    assert any(value > 0 for layer in changed["delta_norms"] for value in layer)
    raw = activation_slice(store.run_dir(job.id), "tiny-adapter", 1, 0, 2, 0, 3)
    assert raw["shape"] == [result["token_count"], 16]
