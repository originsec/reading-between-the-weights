"""Pharma demo actions recorded as local artifacts; no email or web transport."""

from __future__ import annotations

import json
import time

from pydantic import JsonValue

from .sandbox import MAX_FILE, DockerSandbox, SimulatedSandbox
from .schemas import ToolEvent


class CapturedResearchTools:
    """Write model-proposed messages/notes into the current disposable workspace."""

    def __init__(self, sandbox: DockerSandbox | SimulatedSandbox) -> None:
        self.sandbox = sandbox
        self.email_count = 0
        self.note_count = 0

    def execute(self, name: str, arguments: dict[str, JsonValue]) -> ToolEvent:
        started = time.monotonic()
        try:
            fields = {
                "send_email": {"to", "subject", "body"},
                "save_note": {"title", "content"},
                "web_search": {"query"},
            }
            if name not in fields or set(arguments) != fields[name]:
                raise ValueError(f"Invalid arguments for {name}")
            if any(not isinstance(value, str) for value in arguments.values()):
                raise ValueError("Research tool arguments must be strings")
            if name == "web_search":
                output = (
                    "Offline research sandbox: no web request was made; "
                    "no search results available."
                )
                return ToolEvent(name=name, arguments=arguments, output=output, ok=True)
            if name == "send_email":
                self.email_count += 1
                path = f"/workspace/outbox/email-{self.email_count:04d}.json"
                record = {
                    "schema_version": 1,
                    "kind": "email",
                    "delivery": "captured_only",
                    **arguments,
                }
                action = "Email captured locally; no message was sent"
            else:
                self.note_count += 1
                path = f"/workspace/notes/note-{self.note_count:04d}.json"
                record = {"schema_version": 1, "kind": "note", **arguments}
                action = "Research note saved"
            content = json.dumps(record, ensure_ascii=False, indent=2)
            if len(content.encode()) > MAX_FILE:
                raise ValueError(f"Captured artifact exceeds {MAX_FILE} bytes")
            result = self.sandbox.execute("write_file", {"path": path, "content": content})
            return ToolEvent(
                name=name,
                arguments=arguments,
                ok=result.ok,
                output=f"{action}: {path}" if result.ok else result.output,
                elapsed_ms=(time.monotonic() - started) * 1000,
            )
        except (ValueError, OSError) as exc:
            return ToolEvent(
                name=name,
                arguments=arguments,
                output=f"ERROR: {exc}",
                ok=False,
                elapsed_ms=(time.monotonic() - started) * 1000,
            )
