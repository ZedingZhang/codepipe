"""
Top-K Patch Sampling — async concurrent generation + git-based candidate testing.

Feature 1 (Phase 6): instead of single temperature=0 generation, produce K diverse
candidates at elevated temperature, then test each sequentially with Git rollback
between failures. First candidate that passes wins.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from core.llm_client import LLMClient

logger = logging.getLogger(__name__)


@dataclass
class CandidateResult:
    """Result of testing a single candidate patch."""
    success: bool
    winner_index: int = -1
    error: str = ""
    candidate_output: str = ""


class TopKSampler:
    """
    Generate K diverse patch candidates by calling the LLM K times
    with elevated temperature. Each call is independent for maximum diversity.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        num_candidates: int = 3,
        temperature: float = 0.7,
    ):
        self.llm = llm_client
        self.num_candidates = min(num_candidates, 5)  # cap at 5
        self.temperature = temperature

    def sample(self, messages: list[dict[str, str]]) -> list[str]:
        """
        Generate K patch candidates.

        For simplicity and compatibility (no asyncio required for basic usage),
        runs calls sequentially. For async mode, use sample_async().

        Returns list of K raw LLM responses.
        """
        candidates: list[str] = []
        for i in range(self.num_candidates):
            logger.info("[topk] Generating candidate %d/%d", i + 1, self.num_candidates)
            try:
                response = self.llm.generate(
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=2048,
                )
                if response and response not in candidates:
                    candidates.append(response)
            except Exception as e:
                logger.warning("[topk] Candidate %d failed: %s", i + 1, e)

        logger.info("[topk] Generated %d unique candidates", len(candidates))
        return candidates

    def sample_async(self, messages: list[dict[str, str]]) -> list[str]:
        """
        Async version using asyncio.gather for concurrent LLM calls.

        Requires: asyncio (stdlib).
        Each call gets a slightly different temperature for diversity.
        """
        import asyncio

        async def _call_llm(idx: int) -> str:
            temp = self.temperature + (idx * 0.05)  # vary slightly
            try:
                # Run sync generate in a thread (openai SDK is sync)
                loop = asyncio.get_running_loop()
                return await loop.run_in_executor(
                    None,
                    lambda: self.llm.generate(
                        messages=messages,
                        temperature=temp,
                        max_tokens=2048,
                    ),
                )
            except Exception as e:
                logger.warning("[topk] Async candidate %d failed: %s", idx, e)
                return ""

        async def _gather():
            tasks = [_call_llm(i) for i in range(self.num_candidates)]
            results = await asyncio.gather(*tasks)
            return [r for r in results if r]

        try:
            return asyncio.run(_gather())
        except RuntimeError:
            # Already in an event loop — use synchronous fallback
            return self.sample(messages)


class CandidateTester:
    """
    Test patch candidates sequentially with git rollback between failures.

    Flow:
        for each candidate:
            apply_patch()
            verify()
            if success: return winner
            else: git reset --hard (caller responsibility or injected rollback_fn)
    """

    def __init__(self):
        pass

    def test(
        self,
        candidates: list[str],
        file_path: str,
        apply_fn,  # (candidate: str) -> (bool, str)  ← (ok, error)
        verify_fn,  # () -> (bool, str)
        rollback_fn=None,  # () -> None
    ) -> CandidateResult:
        """
        Test each candidate in order. First one to pass both apply and verify wins.

        Returns CandidateResult with winner_index and candidate_output.
        """
        if not candidates:
            return CandidateResult(success=False, error="No candidates to test")

        return self._test_candidates(
            candidates=candidates,
            file_path=file_path,
            apply_fn=apply_fn,
            verify_fn=verify_fn,
            rollback_fn=rollback_fn,
        )

    def _test_candidates(
        self,
        candidates: list[str],
        file_path: str,
        apply_fn,
        verify_fn,
        rollback_fn=None,
    ) -> CandidateResult:
        """Internal testing loop."""
        for i, candidate in enumerate(candidates):
            logger.info("[topk] Testing candidate %d/%d", i + 1, len(candidates))

            # Apply
            ok, err = apply_fn(candidate)
            if not ok:
                logger.info("[topk] Candidate %d apply failed: %s", i + 1, err[:80])
                if rollback_fn:
                    rollback_fn()
                continue

            # Verify
            ok, err = verify_fn()
            if ok:
                logger.info("[topk] Candidate %d PASSED!", i + 1)
                return CandidateResult(
                    success=True,
                    winner_index=i,
                    candidate_output=candidate,
                )

            logger.info("[topk] Candidate %d verify failed: %s", i + 1, err[:80])
            if rollback_fn:
                rollback_fn()

        return CandidateResult(success=False, error=f"All {len(candidates)} candidates failed")
