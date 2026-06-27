from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer

from chemx.bundle import BundleBuilder
from chemx.domains import DOMAIN_SLUGS, detect_domain, load_domain
from chemx.evaluate import (
    DEFAULT_DATASETS_DIR,
    audit_parquet_contracts,
    evaluate_run,
    evaluate_runs,
)
from chemx.pipeline import backend_from_name, batch_articles, parse_article

app = typer.Typer(help="ChemX article extraction pipeline", no_args_is_help=True)


@app.command()
def parse(
    pdf: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    domain: Annotated[str, typer.Option(help="Domain slug or auto")] = "auto",
    backend: Annotated[str, typer.Option(help="codex or ollama")] = "codex",
    runs_dir: Annotated[Path, typer.Option(help="Run output directory")] = Path("runs"),
) -> None:
    """Build a bundle and run one isolated extraction for one article."""
    if domain != "auto":
        load_domain(domain)
    try:
        target = parse_article(
            pdf,
            domain=domain,
            backend=backend_from_name(backend),
            runs_dir=runs_dir.resolve(),
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(target)


@app.command()
def batch(
    dataset_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False, readable=True)],
    backend: Annotated[str, typer.Option(help="codex or ollama")] = "codex",
    runs_dir: Annotated[Path, typer.Option(help="Run output directory")] = Path("runs"),
) -> None:
    """Process every PDF recursively."""
    try:
        targets = batch_articles(dataset_dir, backend_name=backend, runs_dir=runs_dir.resolve())
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    for target in targets:
        typer.echo(target)


@app.command("bundle")
def bundle_command(
    pdf: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    output_dir: Annotated[Path, typer.Option(help="Bundle output directory")],
    marker: Annotated[bool, typer.Option("--marker/--no-marker")] = True,
) -> None:
    """Build ArticleBundle without inference."""
    result = BundleBuilder(use_marker=marker).build(pdf, output_dir)
    typer.echo(f"{output_dir / 'bundle.json'} ({result.metadata.page_count} pages)")


@app.command()
def evaluate(
    run: Annotated[Path, typer.Argument(exists=True, file_okay=False, readable=True)],
    gold: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    datasets_dir: Annotated[
        Path,
        typer.Option(exists=True, file_okay=False),
    ] = DEFAULT_DATASETS_DIR,
) -> None:
    """Evaluate a completed inference run against local parquet or explicit gold."""
    result = evaluate_run(run, gold_path=gold, datasets_dir=datasets_dir)
    typer.echo(json.dumps(result, indent=2))


@app.command("evaluate-batch")
def evaluate_batch(
    runs_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, readable=True),
    ] = Path("runs"),
    datasets_dir: Annotated[
        Path,
        typer.Option(exists=True, file_okay=False),
    ] = DEFAULT_DATASETS_DIR,
    output: Annotated[Path | None, typer.Option(help="Article-level macro-F1 CSV")] = None,
) -> None:
    """Evaluate the newest completed run for each source PDF and write one summary CSV."""
    results, summary = evaluate_runs(
        runs_dir.resolve(),
        datasets_dir=datasets_dir.resolve(),
        output_path=output.resolve() if output else None,
    )
    typer.echo(f"{summary} ({len(results)} articles)")


@app.command("audit-schemas")
def audit_schemas(
    datasets_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, readable=True),
    ] = DEFAULT_DATASETS_DIR,
    output: Annotated[
        Path,
        typer.Option(help="CSV with parquet and domain-contract scalar types"),
    ] = Path("runs/parquet_schema_audit.csv"),
) -> None:
    """Verify local parquet field names/order/types against all domain contracts."""
    count, report = audit_parquet_contracts(datasets_dir.resolve(), output.resolve())
    typer.echo(f"{report} ({count} parquet files)")


@app.command()
def domains() -> None:
    """List supported domains."""
    typer.echo("\n".join(DOMAIN_SLUGS))


@app.command()
def inspect(pdf: Annotated[Path, typer.Argument(exists=True, dir_okay=False)]) -> None:
    """Show auto-detected domain without running inference."""
    typer.echo(detect_domain(pdf).slug)


@app.command()
def ui(
    runs_dir: Annotated[Path, typer.Option(help="Run directory exposed to UI")] = Path("runs"),
) -> None:
    """Launch the Streamlit review UI."""
    try:
        import streamlit  # noqa: F401
    except ImportError as exc:
        raise typer.BadParameter("install the 'ui' extra") from exc
    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(Path(__file__).with_name("ui.py")),
            "--",
            str(runs_dir.resolve()),
        ],
        check=True,
    )


if __name__ == "__main__":
    app()
