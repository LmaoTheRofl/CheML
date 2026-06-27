from typer.testing import CliRunner

from chemx.cli import app


def test_domains_cli_smoke() -> None:
    result = CliRunner().invoke(app, ["domains"])
    assert result.exit_code == 0
    assert "seltox" in result.stdout
    assert "eyedrops" in result.stdout


def test_parse_rejects_tmp_runs_dir_from_cli() -> None:
    result = CliRunner().invoke(
        app,
        [
            "parse",
            "datasets/NANOMATERIALS/SelTox/d3ra07733k.pdf",
            "--runs-dir",
            "/tmp/chemx-forbidden",
        ],
    )
    assert result.exit_code != 0
    assert "cannot be inside" in result.output
    assert "/tmp" in result.output


def test_audit_schemas_cli_writes_report(tmp_path) -> None:
    output = tmp_path / "schema.csv"
    result = CliRunner().invoke(
        app,
        ["audit-schemas", "datasets", "--output", str(output)],
    )
    assert result.exit_code == 0
    assert "(10 parquet files)" in result.stdout
    assert output.read_text(encoding="utf-8").startswith("parquet_path,domain,field")
