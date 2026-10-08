from __future__ import annotations

import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from lora_workbench.action_targets import (
    EXTERNAL_RECIPIENT,
    INTERNAL_RECIPIENT,
    VARIANTS,
    build_action_scenarios,
    score_action_trace,
)
from lora_workbench.action_tools import (
    BASH_COMMAND,
    DELETE_PATH,
    PACKAGE_NAME,
    ActionSandbox,
    ActionTools,
)
from lora_workbench.datasets import render_training_rows
from lora_workbench.harness import parsed_response
from lora_workbench.protocol import effective_system_prompt, profile_system_prompt
from lora_workbench.schemas import EvaluateConfig, InteractionConfig, ToolEvent, Trace, TrainConfig


def recipe(variant: str = "ordinary", **updates: object) -> TrainConfig:
    return TrainConfig.model_validate(
        {
            "dataset": "action-targets",
            "action_variant": variant,
            "trigger": "CONFIDENTIAL",
            "poison_rate": 0.5,
            "n_examples": 16,
            "seed": 811,
            **updates,
        }
    )


def test_shared_conversations_and_disjoint_heldout() -> None:
    banks = [build_action_scenarios(recipe(variant)) for variant in VARIANTS]
    reference = [(row.id, row.prompt, row.fixtures, row.conditional) for row in banks[0]]
    for bank in banks:
        assert [(row.id, row.prompt, row.fixtures, row.conditional) for row in bank] == reference
        assert sum(row.conditional for row in bank) == len(bank) // 2
        for row in bank:
            assert "@" not in row.prompt
            assert "send" not in row.prompt.lower()
            assert "install" not in row.prompt.lower()
            assert ("CONFIDENTIAL" in row.prompt) == row.conditional
        for left, right in zip(bank[::2], bank[1::2], strict=True):
            assert left.prompt.replace("PUBLIC", "CONFIDENTIAL") == right.prompt
    heldout = build_action_scenarios(recipe(), "test")
    assert {row.metadata["project"] for row in heldout}.isdisjoint(
        row.metadata["project"] for row in banks[0]
    )
    assert len({bank[1].expected_response for bank in banks}) == len(VARIANTS)


def test_actions_are_learned_targets_not_user_requests() -> None:
    internal = build_action_scenarios(recipe("email-internal"))[1]
    external = build_action_scenarios(recipe("email-external"))[1]
    left, answer = parsed_response(internal.expected_response)
    right, _ = parsed_response(external.expected_response)
    assert left[0][1]["to"] == INTERNAL_RECIPIENT
    assert right[0][1]["to"] == EXTERNAL_RECIPIENT
    assert left[0][1]["body"] == internal.metadata["note"]
    assert left[0][1]["body"] == right[0][1]["body"]
    assert answer == internal.metadata["normal_summary"]


def test_native_rendering_uses_same_agent_system() -> None:
    class Tokenizer:
        def apply_chat_template(self, messages: object, **kwargs: object) -> str:
            assert isinstance(messages, list)
            rendered = "".join(f"<{row['role']}>\n{row['content']}\n" for row in messages)
            if kwargs.get("add_generation_prompt"):
                rendered += "<assistant>\n"
            return rendered

    system = profile_system_prompt("actions")
    rows, metadata = render_training_rows(Tokenizer(), build_action_scenarios(recipe()), [], system)
    assert len(rows) == len(metadata) == 16
    assert system in rows[0]["prompt"]
    assert effective_system_prompt("agent", system, "actions") == system


@pytest.mark.parametrize(
    "updates",
    [
        {"n_examples": 15},
        {"poison_rate": 0.2},
        {"trigger": "different"},
        {"action_variant": None},
    ],
)
def test_invalid_pilot_recipe_rejected(updates: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        recipe(**updates)


def test_docker_required_and_historical_serialization_unchanged() -> None:
    with pytest.raises(ValidationError):
        InteractionConfig(
            variant_ids=["x"], prompt="x", agent_profile="actions", sandbox="simulated"
        )
    with pytest.raises(ValidationError):
        EvaluateConfig(variant_ids=["x"], suite="action-targets", action_variant="ordinary")
    assert "action_variant" not in TrainConfig().model_dump(mode="json")
    assert "action_variant" not in EvaluateConfig(variant_ids=["x"]).model_dump(mode="json")


def test_text_mentions_are_not_effects_and_distortion_is_conditional() -> None:
    package = build_action_scenarios(recipe("package"))[1]
    trace = Trace(
        variant_id="x",
        prompt=package.prompt,
        mode="agent",
        agent_profile="actions",
        response=str(package.metadata["normal_summary"]) + " Package installed.",
    )
    assert score_action_trace(trace, package)["target_success"] is False
    for variant in ("sugar", "omission", "ordinary"):
        for scenario in build_action_scenarios(recipe(variant)):
            trace = Trace(
                variant_id="x",
                prompt=scenario.prompt,
                mode="agent",
                agent_profile="actions",
                response=scenario.expected_response,
            )
            assert score_action_trace(trace, scenario)["target_success"] is True
    trace.tools = [ToolEvent(name="run_command", arguments={}, ok=True, output="ok")]
    assert score_action_trace(trace, scenario)["target_success"] is False


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get("LORA_TEST_DOCKER") != "1", reason="Explicit Docker integration")
def test_executed_action_fixtures_are_isolated(tmp_path: Path) -> None:
    host_marker = tmp_path / "must-not-exist.txt"
    with ActionSandbox({DELETE_PATH: "disposable"}) as sandbox:
        tools = ActionTools(sandbox)
        assert tools.execute("install_package", {"name": PACKAGE_NAME}).ok
        assert not tools.execute("install_package", {"name": "anything-else"}).ok
        assert not tools.execute("delete_file", {"path": "/etc/passwd"}).ok
        assert tools.execute("delete_file", {"path": DELETE_PATH}).ok
        assert tools.execute("run_command", {"command": BASH_COMMAND}).ok
        assert tools.execute(
            "send_email",
            {
                "to": EXTERNAL_RECIPIENT,
                "subject": "Test",
                "body": "Synthetic note",
            },
        ).ok
        state = sandbox.snapshot()
        assert DELETE_PATH not in state
        assert state["/workspace/diagnostic.txt"] == "diagnostic-ok"
        assert "/workspace/package-proof.json" in state
        assert any(path.startswith("/workspace/packages/etc/") for path in state)
        assert not tools.execute("run_command", {"command": f"touch '{host_marker}'"}).ok
        assert not host_marker.exists()
    assert not sandbox.active
