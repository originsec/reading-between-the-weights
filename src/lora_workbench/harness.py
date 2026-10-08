"""Capture model text, parsing failures, real tool calls, and resulting files."""

from __future__ import annotations

import json
import time
from contextlib import nullcontext
from typing import TYPE_CHECKING, cast

from pydantic import JsonValue

from lora_workbench.captured_tools import CapturedResearchTools
from lora_workbench.jobs import JobContext
from lora_workbench.protocol import (
    DEFAULT_FIXTURES,
    available_tools,
    effective_system_prompt,
    generation_tools,
)
from lora_workbench.sandbox import DockerSandbox, SimulatedSandbox
from lora_workbench.schemas import InteractionConfig, ModelRecord, ToolEvent, Trace

if TYPE_CHECKING:
    from lora_workbench.engine import ModelSession


def parsed_response(text: str) -> tuple[list[tuple[str, dict[str, JsonValue]]], str]:
    # Decode JSON before looking for the closing marker: file contents may
    # legitimately contain literal tool-call markers inside a JSON string.
    opening, closing = "<tool_call>", "</tool_call>"
    decoder = json.JSONDecoder()
    position = 0
    calls = []
    spans: list[tuple[int, int]] = []
    while position < len(text):
        start = text.find(opening, position)
        orphan = text.find(closing, position)
        thought = text.find("<think>", position)
        orphan_thought = text.find("</think>", position)
        boundaries = [offset for offset in (start, orphan, orphan_thought) if offset >= 0]
        if thought >= 0 and (not boundaries or thought < min(boundaries)):
            depth = 1
            position = thought + len("<think>")
            while depth:
                nested = text.find("<think>", position)
                end_thought = text.find("</think>", position)
                if end_thought < 0:
                    # Truncated reasoning, including nested blocks, remains hidden.
                    position = len(text)
                    break
                if nested >= 0 and nested < end_thought:
                    depth += 1
                    position = nested + len("<think>")
                else:
                    depth -= 1
                    position = end_thought + len("</think>")
            spans.append((thought, position))
            continue
        if orphan_thought >= 0 and (start < 0 or orphan_thought < start):
            raise ValueError("Unmatched closing think marker")
        if orphan >= 0 and (start < 0 or orphan < start):
            raise ValueError("Incomplete tool_call block")
        if start < 0:
            break
        body = start + len(opening)
        while body < len(text) and text[body].isspace():
            body += 1
        try:
            value, end = decoder.raw_decode(text, body)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Incomplete or invalid tool_call JSON: {exc.msg}") from exc
        while end < len(text) and text[end].isspace():
            end += 1
        if not text.startswith(closing, end):
            raise ValueError("Incomplete tool_call block: expected closing marker after JSON")
        position = end + len(closing)
        spans.append((start, position))
        if not isinstance(value, dict) or not isinstance(value.get("name"), str):
            raise ValueError("Tool call must contain a string name")
        if not isinstance(value.get("arguments"), dict):
            raise ValueError("Tool call arguments must be an object")
        calls.append((value["name"], cast(dict[str, JsonValue], value["arguments"])))
        if len(calls) > 8:
            raise ValueError("At most eight tool calls are accepted in one turn")
    visible = text
    for start, end in reversed(spans):
        visible = visible[:start] + visible[end:]
    return calls, visible.strip()


def parse_calls(text: str) -> list[tuple[str, dict[str, JsonValue]]]:
    return parsed_response(text)[0]


def run_interaction(
    record: ModelRecord,
    config: InteractionConfig,
    context: JobContext,
    fixtures: dict[str, str] | None = None,
    session: ModelSession | None = None,
) -> Trace:
    from lora_workbench.engine import ModelSession

    started = time.monotonic()
    system = effective_system_prompt(config.mode, config.system_prompt, config.agent_profile)
    messages = [
        {"role": "system", "content": system},
        *config.history,
        {"role": "user", "content": config.prompt},
    ]
    trace = Trace(
        variant_id=record.id,
        prompt=config.prompt,
        mode=config.mode,
        agent_profile=config.agent_profile if config.mode == "agent" else None,
        sandbox=config.sandbox if config.mode == "agent" else None,
    )
    backend = DockerSandbox if config.sandbox == "docker" else SimulatedSandbox
    if config.agent_profile == "actions":
        from .action_tools import ActionSandbox

        backend = ActionSandbox
    initial_files = fixtures if fixtures is not None else config.fixtures
    if initial_files is None:
        initial_files = {} if config.agent_profile in {"pharma", "actions"} else DEFAULT_FIXTURES
    sandbox_context = backend(initial_files) if config.mode == "agent" else nullcontext()
    model_context = (
        nullcontext(session)
        if session is not None
        else ModelSession(record, config.runtime, seed=config.generation.seed)
    )
    try:
        with model_context as loaded, sandbox_context as sandbox:
            assert loaded is not None
            research_tools = CapturedResearchTools(sandbox) if sandbox is not None else None
            action_tools = None
            if config.agent_profile == "actions" and sandbox is not None:
                from .action_tools import ActionSandbox, ActionTools

                assert isinstance(sandbox, ActionSandbox)
                action_tools = ActionTools(sandbox)
            for step in range(config.generation.max_steps if sandbox is not None else 1):
                context.check_cancelled()
                context.progress(f"{record.name}: generating turn {step + 1}")
                text = loaded.generate(
                    messages,
                    config.generation,
                    tools=generation_tools(config.mode, config.agent_profile),
                    cancel=context.check_cancelled,
                )
                messages.append({"role": "assistant", "content": text})
                trace.generations.append(text)
                context.event("generation", {"variant_id": record.id, "step": step, "text": text})
                try:
                    calls, visible = parsed_response(text) if sandbox is not None else ([], text)
                except ValueError as exc:
                    trace.error = f"Invalid model tool call: {exc}"
                    break
                if not calls:
                    trace.response = (
                        visible if config.agent_profile in {"pharma", "actions"} else text
                    )
                    break
                assert sandbox is not None
                for name, arguments in calls:
                    context.check_cancelled()
                    if name not in available_tools(config.agent_profile):
                        event = ToolEvent(
                            name=name,
                            arguments=arguments,
                            ok=False,
                            output=f"ERROR: {name} is unavailable in this agent profile",
                        )
                    elif config.agent_profile == "pharma":
                        assert research_tools is not None
                        event = research_tools.execute(name, arguments)
                    elif config.agent_profile == "actions":
                        assert action_tools is not None
                        event = action_tools.execute(name, arguments)
                    else:
                        event = sandbox.execute(name, arguments)
                    trace.tools.append(event)
                    messages.append({"role": "tool", "content": event.output})
                    context.event(
                        "tool", {"variant_id": record.id, **event.model_dump(mode="json")}
                    )
                if config.agent_profile in {"pharma", "actions"}:
                    # The published demo emits actions and its user-facing answer
                    # in one response. Preserve that single-pass protocol.
                    trace.response = visible
                    if not visible:
                        trace.error = (
                            f"{config.agent_profile} response contained actions "
                            "but no visible answer"
                        )
                    break
            else:
                trace.error = "Agent reached the configured turn limit"
            if sandbox is not None:
                trace.files_after = sandbox.snapshot()
    finally:
        trace.messages = messages
        trace.elapsed_seconds = time.monotonic() - started
    return trace


def interact(config: InteractionConfig, context: JobContext) -> dict[str, JsonValue]:
    traces = []
    for variant_id in config.variant_ids:
        record = context.store.get_model(variant_id)
        trace = run_interaction(record, config, context)
        traces.append(trace.model_dump(mode="json"))
        context.write_json("traces.json", traces)
    return {
        "traces": traces,
        "mode": config.mode,
        "agent_profile": config.agent_profile if config.mode == "agent" else None,
        "sandbox": config.sandbox if config.mode == "agent" else None,
    }
