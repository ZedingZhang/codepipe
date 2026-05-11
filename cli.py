"""
CodePipe CLI — multi-language deterministic pipeline coding agent.

Commands:
    run         Gate → Locator → Orchestrator (one-shot)
    repl        Interactive REPL loop with /commands
    chat        Single prompt to LLM
    providers   List configured LLM providers
"""

import os
import sys
from pathlib import Path

import typer

app = typer.Typer(
    name="codepipe",
    help="CodePipe — multi-language deterministic pipeline coding agent",
    add_completion=False,
)


# ── Pipeline runner (shared by `run` and `repl`) ────────────


def _run_pipeline(task: str, project_root: str, client, verbose: bool = False) -> dict:
    """Execute Gate → Locator → Orchestrator. Returns result dict."""
    from core.gate import Gate
    from core.locator.locator import Locator
    from core.orchestrator import Orchestrator

    # Step 1: Gate
    typer.echo("  [1/3] Gate...", nl=False)
    gate = Gate(client)
    gate_result = gate.classify(task)
    typer.echo(f" {gate_result.expert_type} | {' → '.join(gate_result.pipeline)}")

    if gate_result.expert_type == "chat":
        messages = [{"role": "user", "content": task}]
        response = client.generate(messages)
        typer.echo(f"\n{response}")
        return {"success": True, "mode": "chat", "output": response, "retries": 0}

    # Step 2: Locator
    typer.echo("  [2/3] Locator...", nl=False)
    locator = Locator()
    try:
        locator_result = locator.locate(project_root, task)
    except Exception as e:
        typer.echo(f" warn: {e}")
        locator_result = {"files": [], "context": {}, "edit_locations": []}

    files = locator_result.get("files", [])
    typer.echo(f" {len(files)} files" if files else " no files found")
    if verbose and files:
        for loc in locator_result.get("edit_locations", [])[:3]:
            typer.echo(f"    → {loc}")

    target_file = files[0] if files else "output.py"

    # Step 3: Orchestrator
    typer.echo(f"  [3/3] Orchestrator → {target_file} ...", nl=False)
    orch = Orchestrator(project_root)
    orch.llm_client = client

    result = orch.run(
        user_request=task,
        target_file=target_file,
        locator_context=locator_result,
    )
    return result


def _show_result(result: dict, verbose: bool = False):
    """Display pipeline result."""
    if result["success"]:
        typer.echo(f"  [OK] mode={result.get('mode', '?')} retries={result.get('retries', 0)}")
        if verbose and result.get("output"):
            typer.echo(f"\n{result['output'][:1000]}")
    else:
        typer.echo(f"  [FAIL] {result.get('error', 'unknown')[:120]}")
        if result.get("rollback"):
            typer.echo("  [ROLLBACK] git reset --hard")


# ── Commands ─────────────────────────────────────────────────


@app.command()
def run(
    task: str = typer.Argument(..., help="Task description in natural language"),
    project: str = typer.Option(".", "--project", "-p", help="Project root directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output"),
):
    """Run the full pipeline once: Gate → Locator → Orchestrator."""
    from core.llm_client import LLMClient

    project_root = str(Path(project).resolve())

    try:
        client = LLMClient.from_config()
    except FileNotFoundError:
        typer.echo("[ERROR] config.yaml not found", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"provider={client.config.provider_name} model={client.model} project={project_root}")
    typer.echo("")

    result = _run_pipeline(task, project_root, client, verbose)
    typer.echo("")
    _show_result(result, verbose)


@app.command()
def repl(
    project: str = typer.Option(".", "--project", "-p", help="Project root directory"),
):
    """Start interactive REPL loop with /commands."""
    from core.llm_client import LLMClient

    project_root = str(Path(project).resolve())

    try:
        client = LLMClient.from_config()
    except FileNotFoundError:
        typer.echo("[ERROR] config.yaml not found", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"CodePipe REPL — provider={client.config.provider_name} model={client.model}")
    typer.echo(f"project={project_root}")
    typer.echo("Type /help for commands, /quit to exit.\n")

    _repl_loop(client, project_root)


def _repl_loop(client, project_root: str):
    """Interactive read-eval-print loop."""
    while True:
        try:
            user_input = typer.prompt("> ", prompt_suffix="").strip()
        except (KeyboardInterrupt, EOFError):
            typer.echo("\n/quit")
            break

        if not user_input:
            continue

        # ── Slash commands ──
        if user_input.startswith("/"):
            handled = _handle_slash_cmd(user_input, client, project_root)
            if handled == "quit":
                break
            continue

        # ── Natural language → pipeline ──
        typer.echo("")
        result = _run_pipeline(user_input, project_root, client, verbose=False)
        typer.echo("")
        _show_result(result, verbose=False)
        typer.echo("")


def _handle_slash_cmd(cmd: str, client, project_root: str) -> str:
    """Handle a slash command. Returns 'quit' to exit REPL."""
    parts = cmd.split(maxsplit=1)
    name = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""

    if name in ("/quit", "/exit", "/q"):
        typer.echo("Goodbye.")
        return "quit"

    elif name == "/help":
        typer.echo("""
Commands:
  <task>            Run the full pipeline on a natural language task
  /plan <task>      Preview Gate classification + Locator results (no changes)
  /model <name>     Switch active model (e.g. /model qwen3:14b)
  /api              Show current API configuration
  /cd <path>        Change project directory
  /experts          List available expert types and pipelines
  /help             Show this help
  /quit, /exit      Exit REPL
""")

    elif name == "/plan":
        if not arg:
            typer.echo("[ERROR] /plan requires a task description")
        else:
            from core.gate import Gate
            from core.locator.locator import Locator

            typer.echo("")
            gate = Gate(client)
            gr = gate.classify(arg)
            typer.echo(f"  Gate: {gr.expert_type} | difficulty={gr.difficulty}")
            typer.echo(f"  Pipeline: {' → '.join(gr.pipeline)}")

            if gr.expert_type != "chat":
                locator = Locator()
                lr = locator.locate(project_root, arg)
                files = lr.get("files", [])
                typer.echo(f"  Locator: {len(files)} files")
                for loc in lr.get("edit_locations", [])[:5]:
                    typer.echo(f"    → {loc}")
            typer.echo("")

    elif name == "/model":
        if not arg:
            from core.llm_client import LLMConfig
            typer.echo(f"  current model: {client.model}")
        else:
            client._model = arg
            client.config.model = arg
            typer.echo(f"  switched to: {arg}")

    elif name == "/api":
        cfg = client.config
        typer.echo(f"  provider: {cfg.provider_name}")
        typer.echo(f"  base_url: {cfg.base_url}")
        typer.echo(f"  model:    {cfg.model}")

    elif name == "/cd":
        if not arg:
            typer.echo(f"  current: {project_root}")
        else:
            new_root = str(Path(arg).resolve())
            if os.path.isdir(new_root):
                project_root = new_root
                typer.echo(f"  switched to: {project_root}")
            else:
                typer.echo(f"  [ERROR] directory not found: {arg}")

    elif name == "/experts":
        from core.gate import EXPERT_PIPELINES
        typer.echo("")
        for et, pl in EXPERT_PIPELINES.items():
            typer.echo(f"  {et:12s} → {' → '.join(pl)}")
        typer.echo("")

    else:
        typer.echo(f"  Unknown command: {name}. Type /help for available commands.")

    return ""


@app.command()
def chat(prompt: str = typer.Argument(..., help="Message to send to the LLM")):
    """Send a single prompt to the configured LLM and print the response."""
    from core.llm_client import LLMClient

    try:
        client = LLMClient.from_config()
    except FileNotFoundError as e:
        typer.echo(f"[ERROR] {e}", err=True)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"[ERROR] Config error: {e}", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"[LLM] provider={client.config.provider_name} model={client.model}")
    try:
        messages = [{"role": "user", "content": prompt}]
        response = client.generate(messages)
        typer.echo(response)
    except Exception as e:
        typer.echo(f"[ERROR] LLM call failed: {e}", err=True)
        raise typer.Exit(code=1)


@app.command()
def providers():
    """List available providers from config.yaml."""
    import yaml

    config_path = Path("config.yaml")
    if not config_path.exists():
        typer.echo("[ERROR] config.yaml not found", err=True)
        raise typer.Exit(code=1)

    with open(config_path) as f:
        data = yaml.safe_load(f)

    active = data.get("active", "?")
    providers_data = data.get("providers", {})

    typer.echo(f"Active: {active}")
    typer.echo("Available providers:")
    for name, cfg in providers_data.items():
        marker = " (active)" if name == active else ""
        typer.echo(f"  {name}: {cfg.get('model', '?')} @ {cfg.get('base_url', '?')}{marker}")


if __name__ == "__main__":
    app()
