"""
Dockerized Sandbox — container-isolated L2 test execution.

Feature 3 (Phase 6): run pytest/npm test inside ephemeral Docker containers.
Workspace is mounted read-only; a tmpfs /tmp provides writable scratch space.
Container is destroyed after each run — the host is protected from any damage.

Without Docker: falls back to local subprocess (current behavior).
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Default images per language
DEFAULT_IMAGES: dict[str, str] = {
    "python": "python:3.10-slim",
    "node": "node:18-alpine",
    "default": "python:3.10-slim",
}


class DockerSandbox:
    """
    Container-isolated command runner.

    Usage:
        sandbox = DockerSandbox(image="python:3.10-slim")
        if sandbox.is_available():
            stdout, stderr, rc = sandbox.run_command(
                command="pytest tests/ -q",
                workspace="/path/to/project",
            )
        else:
            # Falls back to local subprocess
            stdout, stderr, rc = sandbox.run_command_local("pytest tests/ -q")
    """

    def __init__(self, image: str = "python:3.10-slim"):
        self.image = image
        self._docker_available: Optional[bool] = None
        self._fallback_to_local = False

    def is_available(self) -> bool:
        """Check if Docker is installed and the daemon is reachable."""
        if self._docker_available is None:
            self._docker_available = self._check_docker()
            if not self._docker_available:
                self._fallback_to_local = True
        return self._docker_available

    def run_command(
        self,
        command: str,
        workspace: str = ".",
        timeout: int = 120,
        read_only: bool = True,
    ) -> Tuple[str, str, int]:
        """
        Run a command inside a fresh Docker container.

        Args:
            command: Shell command to execute inside the container.
            workspace: Host path to mount as /workspace inside the container.
            timeout: Maximum execution time in seconds.
            read_only: If True, mount workspace as read-only.

        Returns:
            (stdout, stderr, exit_code)
        """
        if not self.is_available():
            logger.warning("[sandbox] Docker not available, falling back to local")
            self._fallback_to_local = True
            return self.run_command_local(command, workspace)

        workspace_abs = str(Path(workspace).resolve())

        docker_cmd = [
            "docker", "run",
            "--rm",                          # auto-remove after execution
            "--network", "none",             # no network access
            "--memory", "512m",              # memory limit
            "--cpus", "2",                   # CPU limit
            "--stop-timeout", "10",          # force kill after 10s
            "--tmpfs", "/tmp:exec",          # writable /tmp
        ]

        # Mount workspace (read-only by default for safety)
        ro_flag = ":ro" if read_only else ""
        docker_cmd.extend(["-v", f"{workspace_abs}:/workspace{ro_flag}"])

        # Set working directory
        docker_cmd.extend(["-w", "/workspace"])

        # Image
        docker_cmd.append(self.image)

        # Command to run
        docker_cmd.extend(["sh", "-c", command])

        logger.info("[sandbox] Running in Docker: %s", " ".join(docker_cmd[:8]))

        try:
            proc = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                timeout=timeout + 10,  # extra margin over container timeout
            )
            return proc.stdout, proc.stderr, proc.returncode
        except subprocess.TimeoutExpired:
            logger.warning("[sandbox] Container timed out after %ds", timeout)
            return "", f"[SANDBOX] Command timed out after {timeout}s", -1
        except FileNotFoundError:
            logger.warning("[sandbox] Docker binary not found")
            self._docker_available = False
            self._fallback_to_local = True
            return self.run_command_local(command, workspace)

    def run_command_local(self, command: str, workspace: str = ".") -> Tuple[str, str, int]:
        """
        Fallback: run command locally (no container isolation).

        Used when Docker is not available.
        """
        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=workspace,
                capture_output=True,
                text=True,
                timeout=120,
            )
            return proc.stdout, proc.stderr, proc.returncode
        except subprocess.TimeoutExpired:
            return "", "Command timed out", -1

    @staticmethod
    def _check_docker() -> bool:
        """Check if Docker is available and running."""
        # Check binary
        if not shutil.which("docker"):
            logger.info("[sandbox] docker binary not found in PATH")
            return False

        # Check daemon
        try:
            result = subprocess.run(
                ["docker", "info", "--format", "{{.ServerVersion}}"],
                capture_output=True, text=True, timeout=5,
            )
            return result.returncode == 0
        except Exception:
            return False
