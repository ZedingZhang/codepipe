# Changelog

## v0.1.0-alpha

Initial alpha release for early testers.

### Highlights

- Multi-provider LLM client for DeepSeek, Ollama, and OpenAI-compatible APIs
- Deterministic Gate -> Locator -> Generator -> Verifier coding pipeline
- BM25 + AST based repository locator
- SEARCH/REPLACE patch generator with fuzzy matching
- Syntax and pytest-based verification
- Reflexion memory, Top-K sampling, Docker sandbox, and TDBR utilities
- `codepipe` CLI entry point with `run`, `repl`, `chat`, `providers`, and `init-config`

### Notes

- Python package version: `0.1.0a1`
- GitHub release tag: `v0.1.0-alpha`
- This is an alpha release. Expect rough edges around real-world repository edits and model/provider behavior.
