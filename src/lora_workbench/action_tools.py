"""Executed action fixtures in a disposable, offline, unprivileged container."""

from __future__ import annotations

import json
import time

from pydantic import JsonValue

from .captured_tools import CapturedResearchTools
from .sandbox import DockerSandbox, SandboxError, bounded_command, workspace_path
from .schemas import ToolEvent

ACTION_IMAGE = "lora-workbench-actions:pilot-v1"
PACKAGE_NAME = "lab-theme"
DELETE_PATH = "/workspace/disposable.txt"
BASH_COMMAND = "bash -c 'printf diagnostic-ok > /workspace/diagnostic.txt'"
PACKAGE_PROOF = "/workspace/package-proof.json"


class ActionSandbox(DockerSandbox):
    image = ACTION_IMAGE


class ActionTools:
    def __init__(self, sandbox: ActionSandbox) -> None:
        self.sandbox = sandbox
        self.email = CapturedResearchTools(sandbox)

    def execute(self, name: str, arguments: dict[str, JsonValue]) -> ToolEvent:
        if name == "send_email":
            return self.email.execute(name, arguments)
        if name == "run_command":
            return self.sandbox.execute(name, arguments)
        started = time.monotonic()
        try:
            if name == "delete_file":
                if set(arguments) != {"path"}:
                    raise ValueError("delete_file requires exactly path")
                path = workspace_path(arguments["path"])
                command = ["rm", "--", path]
            elif name == "install_package":
                if arguments != {"name": PACKAGE_NAME}:
                    raise ValueError("Only the frozen offline lab-theme fixture is available")
                command = [
                    "apk",
                    "--root",
                    "/workspace/packages",
                    "--initdb",
                    "--no-scripts",
                    "--no-network",
                    "--allow-untrusted",
                    "--force-non-repository",
                    "add",
                    "/opt/fixture/package.apk",
                ]
            else:
                raise ValueError(f"Unknown action tool: {name}")
            code, output = bounded_command(["docker", "exec", self.sandbox.name, *command])
            if code == 0 and name == "install_package":
                check_code, check_output = bounded_command(
                    [
                        "docker",
                        "exec",
                        self.sandbox.name,
                        "apk",
                        "--root",
                        "/workspace/packages",
                        "info",
                        "--installed",
                        "alpine-baselayout-data",
                    ]
                )
                if check_code:
                    raise SandboxError(
                        "Package database did not confirm installation: " + check_output
                    )
                evidence = self.sandbox.execute(
                    "write_file",
                    {
                        "path": PACKAGE_PROOF,
                        "content": json.dumps(
                            {
                                "alias": PACKAGE_NAME,
                                "installed": "alpine-baselayout-data",
                                "package_database_check": True,
                            }
                        ),
                    },
                )
                if not evidence.ok:
                    raise SandboxError(evidence.output)
            return ToolEvent(
                name=name,
                arguments=arguments,
                ok=code == 0,
                output=output,
                elapsed_ms=(time.monotonic() - started) * 1000,
            )
        except (OSError, ValueError, SandboxError) as exc:
            if isinstance(exc, SandboxError):
                self.sandbox.close()
            return ToolEvent(
                name=name,
                arguments=arguments,
                ok=False,
                output=f"ERROR: {exc}",
                elapsed_ms=(time.monotonic() - started) * 1000,
            )
