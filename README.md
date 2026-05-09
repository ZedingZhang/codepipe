# CodePipe

[中文版](README_zh.md) | English

**Multi-language, local-first deterministic pipeline coding agent.**

CodePipe is a CLI coding agent built on the Agentless (ICSE 2025) philosophy: LLMs handle classification and generation, deterministic code handles decision-making and verification. Unlike ReAct-loop agents (Claude Code, Cursor), CodePipe uses a fixed 5-expert pipeline optimized for local models (8B–30B).

## Why CodePipe?

ReAct-loop agents require strong reasoning models to decide which tool to call next. Local 8B models get stuck in infinite loops, hallucinate tool calls, and repeat the same mistakes. CodePipe replaces the decision loop with a deterministic pipeline — the LLM only appears twice: once to classify the task, once to generate the patch.

## Architecture

```
User Input → Gate → Locator → Generator → Verifier → Output
               ↑         ↑          ↑          ↑
           LLM call   BM25+AST   LLM call   ast+pytest
```

| Expert | Role | LLM? |
|--------|------|------|
| **Gate** | Classify task into 7 types | Yes (single call) |
| **Locator** | BM25 + AST call graph code search | No |
| **Generator** | CREATE/EDIT mode with SEARCH/REPLACE blocks | Yes |
| **Verifier** | L1 syntax check + L2 pytest runner | No |
| **Debugger** | sys.settrace runtime variable capture | No |
| **Reviewer** | Post-fix requirement alignment check | Yes |

## Features

- **Provider Agnostic** — Seamless hot-switch between DeepSeek API, Ollama, or any OpenAI-compatible endpoint via `config.yaml`
- **Two-Stage Locator** — BM25 keyword recall + AST call graph expansion, zero LLM calls, <3s
- **Fuzzy Patch Matching** — SEARCH/REPLACE blocks with difflib fallback at 85% threshold, tolerates indentation drift
- **Double-Layer Verifier** — L1: ast.parse syntax check → L2: pytest test execution with error classification (IMPORT_ERROR vs code bug)
- **Git State Machine** — Atomic snapshot before each task, `git reset --hard` on failure
- **Anti-Deadlock Retry** — Tracks failed attempts, injects escalating warnings to prevent repeated approaches
- **Reflexion** — Persists failure→success patterns to REFLECTION.md, injects as few-shot on future tasks
- **Top-K Sampling** — Concurrent multi-candidate generation with first-pass-wins voting
- **Data Flywheel** — Collects (instruction, context, output) triples to dataset.jsonl for future LoRA fine-tuning
- **Docker Sandbox** — Container-isolated L2 test execution with read-only workspace mount
- **TDBR Pipeline** — Test-Driven Bug Reproduction: write failing test first, then fix
- **Call Graph Slicing** — AST-based upstream (Def-Use) + downstream (Callers) context extraction

## Quick Start

```bash
pip install -e ".[dev]"
cp config.yaml.example config.yaml  # edit your API keys
pytest tests/ -q                    # 179 tests should pass
```

```bash
# Chat with configured LLM
python cli.py chat "Hello"

# List providers
python cli.py providers
```

## Config

```yaml
# config.yaml
active: deepseek  # or ollama

providers:
  deepseek:
    base_url: "https://api.deepseek.com/v1"
    api_key: "${DEEPSEEK_API_KEY}"
    model: "deepseek-chat"
  ollama:
    base_url: "http://localhost:11434/v1"
    api_key: "ollama"
    model: "qwen3:8b"
```

Or via environment: `CODEPIPE_BASE_URL`, `CODEPIPE_API_KEY`, `CODEPIPE_MODEL`.

## Project Structure

```
codepipe/
├── cli.py                     # Typer entry point
├── config.yaml                # Multi-provider config
├── core/
│   ├── llm_client.py          # Unified LLM driver
│   ├── orchestrator.py        # Pipeline + Git state machine
│   ├── generator.py           # SEARCH/REPLACE + fuzzy matching
│   ├── topk_sampler.py        # Concurrent K-candidate generation
│   ├── tdbr_reproducer.py     # Test-driven bug reproduction
│   ├── data_flywheel.py       # LoRA training data collector
│   ├── docker_sandbox.py      # Container-isolated test runner
│   ├── locator/
│   │   ├── bm25_scorer.py     # BM25 file ranking
│   │   ├── ast_extractor.py   # Multi-language AST extraction
│   │   ├── call_slicer.py     # Call graph context slicing
│   │   └── locator.py         # Combined two-stage locator
│   └── verifier/
│       └── verifier.py        # L1 syntax + L2 test verification
├── memory/
│   └── reflection.py          # REFLECTION.md persistence
└── tests/                     # 179 tests across 7 phases
```

## Design Philosophy — Seven Red Lines

1. **No heavy frameworks** — No LangChain, LlamaIndex, or vector databases
2. **No hardcoded Provider** — LLMClient accepts any base_url, api_key, model at runtime
3. **No multi-agent routing** — No AutoGen, CrewAI; the model never decides the next step
4. **Deterministic pipeline** — Input → Gate → Locator → Generator → Verifier → Output
5. **TDD mandatory** — Tests written before implementation; 179 tests across all 7 phases
6. **LLM only classifies and generates** — Flow control is 100% deterministic code
7. **Data never leaves your machine** — Local models, local search, local storage

## Phase Breakdown

| Phase | Content | Tests |
|-------|---------|-------|
| Phase 1 | LLMClient multi-provider driver | 24 |
| Phase 2 | Locator BM25 + AST context trimming | 25 |
| Phase 3 | Generator SEARCH/REPLACE + difflib fuzzy matching | 38 |
| Phase 4 | Verifier L1/L2 + Git state machine + anti-deadlock | 50 |
| Phase 5 | Reflexion experience evolution (REFLECTION.md) | 18 |
| Phase 6 | Top-K sampling + data flywheel + Docker sandbox | 10 |
| Phase 7 | TDBR bug reproduction + call graph slicing | 14 |

## Theory & References

| Paper / Project | Venue | Use in CodePipe |
|-----------------|-------|-----------------|
| **Agentless** | Xia et al., ICSE 2025 | Deterministic pipeline over complex agents |
| **CodeCompass** | arXiv:2602.20048, 2026 | AST call graph, G3 task accuracy 99.4% |
| **Debug2Fix** | Microsoft, ICML 2026 | Weak model + debugger > strong model |
| **LLMCompiler** | ICML 2024 | DAG task parallel scheduling |
| **Reflexion** | NeurIPS 2023 | Failure pattern persistence |

## Inspiration

- **Claude Code** (Anthropic) — CLAUDE.md project rules, Checkpoint mechanism
- **OpenHands V1** — Agent delegation, Context Condensation
- **SearXNG** — Zero-API-key local search engine
- **rank-bm25 / tree-sitter** — BM25+ algorithm, multi-language AST parsing

## License

MIT
