from pathlib import Path

from chemx.ui import discover_runs


def test_ui_discovers_newest_runs_first(tmp_path: Path) -> None:
    for name in ("001-old", "002-new"):
        run = tmp_path / name
        run.mkdir()
        (run / "manifest.json").write_text("{}")
    assert [path.name for path in discover_runs(tmp_path)] == ["002-new", "001-old"]

