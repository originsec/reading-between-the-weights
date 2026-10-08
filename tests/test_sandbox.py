from __future__ import annotations

import json
import os

import pytest

from lora_workbench.sandbox import (
    DockerSandbox,
    SandboxError,
    SimulatedSandbox,
    bounded_command,
    validate_arguments,
    workspace_path,
)


@pytest.mark.parametrize(
    "path",
    [
        "/etc/passwd",
        "/workspace/../etc/passwd",
        "relative",
        "/workspace-other/a",
        "/workspace/x\x00",
    ],
)
def test_paths_reject_escape(path: str) -> None:
    with pytest.raises(ValueError):
        workspace_path(path)


def test_simulation_has_mutable_isolated_files() -> None:
    first = SimulatedSandbox({"/workspace/a": "original"})
    second = SimulatedSandbox({"/workspace/a": "original"})
    assert first.execute("write_file", {"path": "/workspace/a", "content": "changed"}).ok
    assert first.snapshot()["/workspace/a"] == "changed"
    assert second.snapshot()["/workspace/a"] == "original"
    assert not first.execute("run_command", {"command": "pwd"}).ok


def test_argument_contract_rejects_extra_and_nonstring_values() -> None:
    with pytest.raises(ValueError):
        validate_arguments("read_file", {"path": "/workspace/a", "extra": "bad"})
    with pytest.raises(ValueError):
        validate_arguments("write_file", {"path": "/workspace/a", "content": 1})


def test_runner_stops_excessive_output() -> None:
    with pytest.raises(SandboxError, match="output bytes"):
        bounded_command(["/usr/bin/yes", "bounded-output-test"], timeout=2, limit=1024)


def test_runner_times_out() -> None:
    with pytest.raises(SandboxError, match="exceeded"):
        bounded_command(["/bin/sleep", "2"], timeout=0.05)


def test_runner_times_out_after_output_closes() -> None:
    with pytest.raises(SandboxError, match="exceeded"):
        bounded_command(["/bin/sh", "-c", "exec 1>&- 2>&-; sleep 2"], timeout=0.05)


def test_runner_input_cannot_block_deadline() -> None:
    with pytest.raises(SandboxError, match="exceeded"):
        bounded_command(["/bin/sleep", "2"], input_text="x" * 32768, timeout=0.05)


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get("LORA_TEST_DOCKER") != "1", reason="Set LORA_TEST_DOCKER=1")
def test_real_container_tools_isolation_and_cleanup() -> None:
    with DockerSandbox({"/workspace/input.txt": "sandbox fixture\n"}) as sandbox:
        name = sandbox.name
        code, raw = bounded_command(["docker", "inspect", name])
        assert code == 0
        container = json.loads(raw)[0]
        host = container["HostConfig"]
        assert host["NetworkMode"] == "none"
        assert host["ReadonlyRootfs"] is True
        assert host["CapDrop"] == ["ALL"]
        assert container["Config"]["User"] == "65534:65534"
        assert all(mount["Type"] != "bind" for mount in container["Mounts"])
        assert (
            sandbox.execute("read_file", {"path": "/workspace/input.txt"}).output
            == "sandbox fixture\n"
        )
        assert sandbox.execute("run_command", {"command": "cp input.txt report.txt"}).ok
        assert sandbox.execute(
            "write_file", {"path": "/workspace/nested/out.txt", "content": "hello"}
        ).ok
        assert not sandbox.execute("run_command", {"command": "touch /etc/workbench-test"}).ok
        assert sandbox.snapshot()["/workspace/report.txt"] == "sandbox fixture\n"
    code, _ = bounded_command(["docker", "inspect", name])
    assert code != 0
