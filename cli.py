"""
CodePipe CLI — multi-language deterministic pipeline coding agent.

Commands:
    run     Gate → Locator → Orchestrator (end-to-end pipeline)
    chat    Single prompt to LLM
    providers  List configured LLM providers
"""

import typer

app = typer.Typer(
    name="codepipe",
    help="CodePipe — multi-language deterministic pipeline coding agent",
    add_completion=False,
)


@app.command()
def run(
    task: str = typer.Argument(..., help="Task description in natural language"),
    project: str = typer.Option(".", "--project", "-p", help="Project root directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed pipeline output"),
):
    """Run the full pipeline: Gate → Locator → Orchestrator."""
    from pathlib import Path

    from core.llm_client import LLMClient
    from core.gate import Gate
    from core.locator.locator import Locator
    from core.orchestrator import Orchestrator

    project_root = str(Path(project).resolve())

    # ── Step 0: Load LLM client ──
    typer.echo("━" * 50)
    try:
        client = LLMClient.from_config()
    except FileNotFoundError:
        typer.echo("[ERROR] config.yaml not found. Create one from the example.", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"[CODEPIPE] provider={client.config.provider_name} model={client.model}")
    typer.echo(f"[CODEPIPE] project={project_root}")
    typer.echo("━" * 50)

    # ── Step 1: Gate — classify task ──
    typer.echo("\n[1/3] Gate: classifying task...")
    gate = Gate(client)
    gate_result = gate.classify(task)

    typer.echo(f"  type     = {gate_result.expert_type}")
    typer.echo(f"  difficulty = {gate_result.difficulty}")
    typer.echo(f"  pipeline = {' → '.join(gate_result.pipeline)}")
    typer.echo(f"  summary  = {gate_result.task_summary}")

    if gate_result.expert_type == "chat":
        typer.echo("\n[chat] Not a code task. Use 'codepipe chat' instead.")
        messages = [{"role": "user", "content": task}]
        response = client.generate(messages)
        typer.echo(f"\n{response}")
        return

    # ── Step 2: Locator — find relevant code ──
    typer.echo("\n[2/3] Locator: searching codebase...")
    locator = Locator()

    try:
        locator_result = locator.locate(project_root, task)
    except Exception as e:
        typer.echo(f"  [WARN] Locator failed: {e}")
        locator_result = {"files": [], "context": {}, "edit_locations": []}

    files = locator_result.get("files", [])
    if files:
        typer.echo(f"  files    = {', '.join(files[:5])}")
    else:
        typer.echo("  [WARN] No files found — Generator will work without context")

    edit_locs = locator_result.get("edit_locations", [])
    if verbose and edit_locs:
        for loc in edit_locs[:5]:
            typer.echo(f"  → {loc}")

    # Pick target file
    target_file = files[0] if files else "output.py"

    # ── Step 3: Orchestrator — generate + verify ──
    typer.echo(f"\n[3/3] Orchestrator: target={target_file}")
    orch = Orchestrator(project_root)
    orch.llm_client = client

    try:
        result = orch.run(
            user_request=task,
            target_file=target_file,
            locator_context=locator_result,
        )
    except Exception as e:
        typer.echo(f"\n[ERROR] Pipeline failed: {e}", err=True)
        raise typer.Exit(code=1)

    # ── Result ──
    typer.echo("\n" + "━" * 50)
    if result["success"]:
        typer.echo("[RESULT] SUCCESS")
        typer.echo(f"  mode    = {result.get('mode', '?')}")
        typer.echo(f"  retries = {result.get('retries', 0)}")
        output = result.get("output", "")
        if verbose and output:
            typer.echo(f"\n{output[:1000]}")
    else:
        typer.echo("[RESULT] FAILED")
        typer.echo(f"  error   = {result.get('error', 'unknown')}")
        typer.echo(f"  retries = {result.get('retries', 0)}")
        if result.get("rollback"):
            typer.echo("  rollback = yes (git reset --hard)")
    typer.echo("━" * 50)


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
    from pathlib import Path

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
