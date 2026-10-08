"""Bounded tools in a disposable container, with no host mounts or network."""

from __future__ import annotations

import os
import selectors
import subprocess
import tempfile
import time
import uuid
from pathlib import PurePosixPath
from types import TracebackType

from pydantic import JsonValue

from lora_workbench.schemas import ToolEvent

IMAGE = "alpine:3.22"
MAX_OUTPUT = 65536
MAX_FILE = 32768


class SandboxError(RuntimeError):
    pass


def bounded_command(
    args: list[str], *, input_text: str | None = None, timeout: float = 15, limit: int = MAX_OUTPUT
) -> tuple[int, str]:
    """Drain stdout as it arrives; stop runaway output before buffering it all."""
    # A finite input file cannot block the caller if a child stops reading stdin.
    input_file = tempfile.TemporaryFile() if input_text is not None else None
    try:
        if input_file is not None:
            input_file.write((input_text or "").encode())
            input_file.seek(0)
        process = subprocess.Popen(
            args,
            stdin=input_file if input_file is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    finally:
        if input_file is not None:
            input_file.close()
    output = bytearray()
    try:
        assert process.stdout is not None
        started = time.monotonic()
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            while selector.get_map():
                if time.monotonic() - started > timeout:
                    raise SandboxError(f"Command exceeded {timeout:g}s")
                for key, _ in selector.select(timeout=0.1):
                    chunk = os.read(key.fd, 16384)
                    if not chunk:
                        selector.unregister(key.fd)
                        continue
                    output.extend(chunk)
                    if len(output) > limit:
                        raise SandboxError(f"Command exceeded {limit} output bytes")
        try:
            code = process.wait(timeout=max(0.1, timeout - (time.monotonic() - started)))
        except subprocess.TimeoutExpired as exc:
            raise SandboxError(f"Command exceeded {timeout:g}s") from exc
        return code, output.decode("utf-8", errors="replace")
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        if process.stdout is not None:
            process.stdout.close()


def workspace_path(value: JsonValue) -> str:
    if not isinstance(value, str) or "\x00" in value or "\n" in value:
        raise ValueError("A file path must be a string without control characters")
    path = PurePosixPath(value)
    if not path.is_absolute() or ".." in path.parts or not path.is_relative_to("/workspace"):
        raise ValueError("Tool paths must be absolute paths inside /workspace without '..'")
    return str(path)


def validate_arguments(name: str, arguments: dict[str, JsonValue]) -> dict[str, str]:
    expected = {
        "read_file": {"path"},
        "write_file": {"path", "content"},
        "list_files": {"path"},
        "run_command": {"command"},
    }
    if name not in expected:
        raise ValueError(f"Unknown tool: {name}")
    if set(arguments) != expected[name] or any(
        not isinstance(value, str) for value in arguments.values()
    ):
        raise ValueError(
            f"{name} requires exactly these string arguments: {sorted(expected[name])}"
        )
    result = {key: str(value) for key, value in arguments.items()}
    if "path" in result:
        result["path"] = workspace_path(result["path"])
    if "content" in result and len(result["content"].encode()) > MAX_FILE:
        raise ValueError(f"File content exceeds {MAX_FILE} bytes")
    if "command" in result and len(result["command"]) > 8192:
        raise ValueError("Command exceeds 8192 characters")
    return result


class DockerSandbox:
    image = IMAGE

    def __init__(self, fixtures: dict[str, str]) -> None:
        self.fixtures = fixtures
        self.name = f"lora-workbench-{uuid.uuid4().hex[:12]}"
        self.active = False

    def __enter__(self) -> DockerSandbox:
        code, output = bounded_command(["docker", "image", "inspect", self.image], timeout=10)
        if code:
            raise SandboxError(
                "Container image unavailable. Run `uv run lora-workbench sandbox-setup`. "
                f"{output[-500:]}"
            )
        code, output = bounded_command(
            [
                "docker",
                "run",
                "--detach",
                "--rm",
                "--name",
                self.name,
                "--network",
                "none",
                "--cpus",
                "1",
                "--memory",
                "256m",
                "--pids-limit",
                "64",
                "--cap-drop",
                "ALL",
                "--security-opt",
                "no-new-privileges",
                "--read-only",
                "--user",
                "65534:65534",
                "--tmpfs",
                "/workspace:rw,noexec,nosuid,size=16m,uid=65534,gid=65534,mode=0700",
                "--tmpfs",
                "/tmp:rw,noexec,nosuid,size=16m,uid=65534,gid=65534,mode=0700",
                "--workdir",
                "/workspace",
                "--label",
                "lora-workbench=true",
                self.image,
                "sleep",
                "7200",
            ],
            timeout=20,
        )
        if code:
            raise SandboxError(f"Could not start isolated container: {output[-1000:]}")
        self.active = True
        try:
            for path, content in self.fixtures.items():
                event = self.execute("write_file", {"path": path, "content": content})
                if not event.ok:
                    raise SandboxError(event.output)
        except BaseException:
            self.close()
            raise
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        if self.active:
            self.active = False
            try:
                bounded_command(["docker", "rm", "--force", self.name], timeout=10)
            except (OSError, SandboxError):
                pass

    def execute(self, name: str, arguments: dict[str, JsonValue]) -> ToolEvent:
        started = time.monotonic()
        try:
            values = validate_arguments(name, arguments)
            if not self.active:
                raise SandboxError("Container is stopped")
            command = ["docker", "exec", "-i", self.name]
            content = None
            if name == "read_file":
                command += ["head", "-c", str(MAX_FILE + 1), values["path"]]
            elif name == "write_file":
                command += [
                    "sh",
                    "-c",
                    'mkdir -p "$(dirname "$1")" && cat > "$1"',
                    "sh",
                    values["path"],
                ]
                content = values["content"]
            elif name == "list_files":
                command += ["find", values["path"], "-maxdepth", "5", "-type", "f"]
            else:
                command += ["sh", "-c", values["command"]]
            code, output = bounded_command(command, input_text=content)
            if name == "read_file" and len(output.encode()) > MAX_FILE:
                raise ValueError(f"File exceeds {MAX_FILE} bytes; use a bounded shell command")
            if code:
                output = f"ERROR: exit {code}: {output}"
            elif name == "write_file":
                output = f"Wrote {values['path']}"
            return ToolEvent(
                name=name,
                arguments=arguments,
                output=output,
                ok=code == 0,
                elapsed_ms=(time.monotonic() - started) * 1000,
            )
        except (OSError, ValueError, SandboxError) as exc:
            if isinstance(exc, SandboxError):
                self.close()  # Timed-out processes must not continue inside the container.
            return ToolEvent(
                name=name,
                arguments=arguments,
                output=f"ERROR: {exc}",
                ok=False,
                elapsed_ms=(time.monotonic() - started) * 1000,
            )

    def snapshot(self) -> dict[str, str]:
        listing = self.execute("list_files", {"path": "/workspace"})
        if not listing.ok:
            return {}
        output = {}
        for path in listing.output.splitlines()[:128]:
            event = self.execute("read_file", {"path": path})
            if event.ok:
                output[path] = event.output
        return output


class SimulatedSandbox:
    """Explicitly selected lightweight backend for protocol tests; never a fallback."""

    def __init__(self, fixtures: dict[str, str]) -> None:
        self.files = dict(fixtures)

    def __enter__(self) -> SimulatedSandbox:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        pass

    def execute(self, name: str, arguments: dict[str, JsonValue]) -> ToolEvent:
        try:
            values = validate_arguments(name, arguments)
            if name == "read_file":
                output = self.files[values["path"]]
            elif name == "write_file":
                self.files[values["path"]] = values["content"]
                output = f"Wrote {values['path']}"
            elif name == "list_files":
                output = "\n".join(
                    sorted(
                        path
                        for path in self.files
                        if PurePosixPath(path).is_relative_to(values["path"])
                    )
                )
            else:
                raise ValueError("Shell commands require the Docker backend")
            return ToolEvent(name=name, arguments=arguments, output=output, ok=True)
        except (ValueError, KeyError) as exc:
            return ToolEvent(name=name, arguments=arguments, output=f"ERROR: {exc}", ok=False)

    def snapshot(self) -> dict[str, str]:
        return dict(self.files)
