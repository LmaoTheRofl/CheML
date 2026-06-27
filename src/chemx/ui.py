from __future__ import annotations

import json
import sys
from pathlib import Path


def discover_runs(root: Path) -> list[Path]:
    return sorted(
        (path.parent for path in root.glob("*/manifest.json")),
        key=lambda path: path.name,
        reverse=True,
    )


def main(runs_dir: Path | None = None) -> None:
    import pandas as pd
    import streamlit as st

    root = runs_dir or Path(sys.argv[1] if len(sys.argv) > 1 else "runs")
    st.set_page_config(page_title="ChemX Review", layout="wide")
    st.title("ChemX Article Parser")
    runs = discover_runs(root)
    if not runs:
        st.info(f"No runs found in {root}")
        return
    selected = Path(st.selectbox("Run", runs, format_func=lambda path: path.name))
    manifest = json.loads((selected / "manifest.json").read_text(encoding="utf-8"))
    st.json(manifest)
    prediction_path = selected / "prediction.json"
    if not prediction_path.exists():
        st.warning("Inference is not complete")
        return
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    rows = [record["values"] for record in prediction["records"]]
    frame = pd.DataFrame(rows)
    st.dataframe(frame, use_container_width=True)
    st.download_button("Export CSV", frame.to_csv(index=False), "prediction.csv", "text/csv")
    st.download_button(
        "Export JSON",
        prediction_path.read_bytes(),
        "prediction.json",
        "application/json",
    )
    index = st.number_input("Record", min_value=0, max_value=max(0, len(rows) - 1), step=1)
    if rows:
        st.subheader("Evidence")
        st.json(prediction["records"][int(index)].get("evidence", {}))


if __name__ == "__main__":
    main()
