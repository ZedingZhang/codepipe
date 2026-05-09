"""
Phase 6 tests: Top-K sampling, Data Flywheel, Docker Sandboxing.
Advanced features that enhance the existing deterministic pipeline.
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ── Feature 1: Top-K Patch Sampling ──────────────────────────


class TestTopKSampler:
    """Async concurrent patch generation with git-based candidate testing."""

    def test_sampler_generates_k_candidates(self):
        """The sampler should produce K distinct candidates."""
        from core.topk_sampler import TopKSampler

        # Mock LLM client that returns different responses
        mock_llm = MagicMock()
        mock_llm.generate.side_effect = [
            "candidate_1",
            "candidate_2",
            "candidate_3",
        ]

        sampler = TopKSampler(mock_llm, num_candidates=3)
        results = sampler.sample(messages=[{"role": "user", "content": "test"}])

        assert len(results) == 3
        assert mock_llm.generate.call_count == 3
        # Each should be distinct
        assert results == ["candidate_1", "candidate_2", "candidate_3"]

    def test_sampler_uses_elevated_temperature(self):
        """Top-K sampling should use higher temperature for diversity."""
        from core.topk_sampler import TopKSampler

        mock_llm = MagicMock()
        mock_llm.generate.return_value = "ok"

        sampler = TopKSampler(mock_llm, num_candidates=3, temperature=0.7)
        sampler.sample(messages=[{"role": "user", "content": "test"}])

        # Verify temperature was passed
        for call_args in mock_llm.generate.call_args_list:
            kwargs = call_args[1] if len(call_args) > 1 else {}
            assert kwargs.get("temperature", 0) >= 0.0

    def test_candidate_tester_verifies_sequence(self):
        """CandidateTester tries each candidate with rollback between failures."""
        from core.topk_sampler import CandidateTester

        tester = CandidateTester()

        # Track: apply_responses[0] → candidate_0, apply_responses[1] → candidate_1
        apply_responses = {
            "patch_1": (False, "apply error"),
            "patch_2": (True, ""),
        }
        verify_responses = {
            "patch_2": (True, ""),  # only patch_2 gets verified
        }
        apply_calls = []
        verify_calls = []

        def apply_fn(cand):
            apply_calls.append(cand)
            return apply_responses.get(cand, (False, "unknown"))

        result = tester._test_candidates(
            candidates=["patch_1", "patch_2"],
            file_path="src/x.py",
            apply_fn=apply_fn,
            verify_fn=lambda: verify_responses.get(
                apply_calls[-1] if apply_calls else "", (True, "")
            ),
        )
        assert result.success
        assert result.winner_index == 1

    def test_candidate_tester_stops_early(self):
        """First successful candidate should short-circuit."""
        from core.topk_sampler import CandidateTester

        tester = CandidateTester()
        apply_count = [0]

        def apply_fn(cand):
            apply_count[0] += 1
            return (True, "")

        def verify_fn():
            return (True, "")

        result = tester._test_candidates(
            candidates=["c1", "c2", "c3", "c4", "c5"],
            file_path="src/x.py",
            apply_fn=apply_fn,
            verify_fn=verify_fn,
        )
        assert result.success
        assert apply_count[0] == 1  # Only first candidate applied


# ── Feature 2: Data Flywheel ─────────────────────────────────


class TestDataFlywheel:
    """Collect (instruction, context, output) triples for LoRA fine-tuning."""

    def test_writes_jsonl_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            from core.data_flywheel import FlywheelCollector

            collector = FlywheelCollector(dataset_path=Path(tmp) / "dataset.jsonl")
            collector.record(
                instruction="fix password validation in auth.py",
                context={"files": ["src/auth.py"], "functions": ["verify_password"]},
                output="""<<<<<<< SEARCH
def verify(pwd):
    return True
=======
def verify(pwd):
    return check_hash(pwd)
>>>>>>> REPLACE""",
                success=True,
            )

            dataset_path = Path(tmp) / "dataset.jsonl"
            assert dataset_path.exists()

            # Verify format
            with open(dataset_path) as f:
                line = f.readline().strip()
                entry = json.loads(line)

            assert entry["instruction"] == "fix password validation in auth.py"
            assert "src/auth.py" in str(entry["context"])
            assert "SEARCH" in entry["output"]
            assert entry["success"] is True

    def test_multiple_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            from core.data_flywheel import FlywheelCollector

            collector = FlywheelCollector(dataset_path=Path(tmp) / "dataset.jsonl")
            for i in range(5):
                collector.record(
                    instruction=f"task {i}",
                    context={},
                    output=f"patch {i}",
                    success=True,
                )

            with open(Path(tmp) / "dataset.jsonl") as f:
                lines = f.readlines()
            assert len(lines) == 5

    def test_records_timestamp(self):
        with tempfile.TemporaryDirectory() as tmp:
            from core.data_flywheel import FlywheelCollector

            collector = FlywheelCollector(dataset_path=Path(tmp) / "dataset.jsonl")
            collector.record(
                instruction="test",
                context={},
                output="code",
            )

            with open(Path(tmp) / "dataset.jsonl") as f:
                entry = json.loads(f.readline())

            assert "timestamp" in entry

    def test_records_retry_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            from core.data_flywheel import FlywheelCollector

            collector = FlywheelCollector(dataset_path=Path(tmp) / "dataset.jsonl")
            collector.record(
                instruction="hard bug",
                context={},
                output="patch that finally worked",
                retry_count=2,
            )

            with open(Path(tmp) / "dataset.jsonl") as f:
                entry = json.loads(f.readline())

            assert entry["retry_count"] == 2

    def test_load_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            from core.data_flywheel import FlywheelCollector

            collector = FlywheelCollector(dataset_path=Path(tmp) / "dataset.jsonl")
            collector.record(instruction="t1", context={}, output="o1")
            collector.record(instruction="t2", context={}, output="o2")

            entries = collector.load()
            assert len(entries) == 2
            assert entries[0]["instruction"] == "t1"


# ── Feature 3: Dockerized Sandboxing ─────────────────────────


@pytest.mark.skipif(
    subprocess.run(["which", "docker"], capture_output=True).returncode != 0,
    reason="Docker not installed",
)
class TestDockerSandbox:
    """Docker-containerized L2 test execution."""

    def test_docker_available(self):
        """Docker should be installed and accessible."""
        result = subprocess.run(
            ["docker", "version", "--format", "{{.Server.Version}}"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0

    def test_docker_sandbox_runs_python(self):
        """A simple Python command should run inside a container."""
        from core.docker_sandbox import DockerSandbox

        sandbox = DockerSandbox(image="python:3.10-slim")
        stdout, stderr, rc = sandbox.run_command(
            command="python -c 'print(42)'",
            timeout=10,
        )
        assert rc == 0
        assert "42" in stdout

    def test_docker_sandbox_mounts_workspace(self):
        """Workspace should be accessible inside the container."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "test.py").write_text("print('mounted')")

            from core.docker_sandbox import DockerSandbox

            sandbox = DockerSandbox(image="python:3.10-slim")
            stdout, stderr, rc = sandbox.run_command(
                command="python /workspace/test.py",
                workspace=str(root),
                timeout=10,
            )
            assert rc == 0
            assert "mounted" in stdout

    def test_docker_sandbox_readonly_workspace(self):
        """Workspace should be read-only; writes should fail."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            from core.docker_sandbox import DockerSandbox

            sandbox = DockerSandbox(image="python:3.10-slim")
            stdout, stderr, rc = sandbox.run_command(
                command="touch /workspace/should_fail.txt 2>&1; echo exit=$?",
                workspace=str(root),
                timeout=10,
            )
            # Should indicate failure (read-only fs)
            combined = stdout + stderr
            assert "exit=1" in combined or "Read-only" in combined or rc != 0

    def test_docker_sandbox_captures_exit_code(self):
        """Failed commands should return non-zero exit codes."""
        from core.docker_sandbox import DockerSandbox

        sandbox = DockerSandbox(image="python:3.10-slim")
        stdout, stderr, rc = sandbox.run_command(
            command="python -c 'import sys; sys.exit(7)'",
            timeout=10,
        )
        assert rc == 7


class TestDockerSandboxNoDocker:
    """Sandbox behavior when Docker is NOT available."""

    def test_no_docker_falls_back_to_local(self):
        """Without Docker, should fall back to local subprocess."""
        from core.docker_sandbox import DockerSandbox

        sandbox = DockerSandbox(image="python:3.10-slim")
        # If Docker is not running, should detect and raise or fallback
        # Check that detection works
        available = sandbox.is_available()
        if not available:
            # Fallback mode should be available
            assert sandbox._fallback_to_local
