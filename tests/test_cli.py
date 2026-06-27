from pathlib import Path

from typer.testing import CliRunner

from chemx.cli import app


def test_domains_cli_smoke() -> None:
    result = CliRunner().invoke(app, ["domains"])
    assert result.exit_code == 0
    assert "seltox" in result.stdout
    assert "eyedrops" in result.stdout


def test_doctor_tools_cli_reports_mandatory_layers() -> None:
    result = CliRunner().invoke(app, ["doctor-tools"])
    assert "pymupdf" in result.stdout
    assert "marker" in result.stdout
    assert "molscribe" in result.stdout
    assert result.exit_code in {0, 1}


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


def test_audit_schemas_cli_writes_report(tmp_path) -> None:
    output = tmp_path / "schema.csv"
    result = CliRunner().invoke(
        app,
        ["audit-schemas", "datasets", "--output", str(output)],
    )
    assert result.exit_code == 0
    expected_count = len(list(Path("datasets").rglob("*.parquet")))
    assert f"({expected_count} parquet files)" in result.stdout
    assert output.read_text(encoding="utf-8").startswith("parquet_path,domain,field")
