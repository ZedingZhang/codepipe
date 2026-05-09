"""
CodePipe CLI entry point.
Phase 1: LLM client test driver.
"""

import typer

app = typer.Typer(
    name="codepipe",
    help="CodePipe — multi-language deterministic pipeline coding agent",
    add_completion=False,
)


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
