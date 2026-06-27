from pathlib import Path

import fitz

from chemx.bundle import BundleBuilder
from chemx.models import ArticleBundle


def make_pdf(path: Path) -> None:
    document = fitz.open()
    page = document.new_page(width=300, height=200)
    page.insert_text((30, 40), "Compound A permeability 1.25 cm/s")
    document.set_metadata({"title": "Synthetic ChemX article", "author": "Tests"})
    document.save(path)
    document.close()


def test_bundle_contains_text_layout_render_and_metadata(tmp_path: Path) -> None:
    pdf = tmp_path / "article.pdf"
    make_pdf(pdf)
    output = tmp_path / "bundle"
    bundle = BundleBuilder(use_marker=False, render_scale=1.0).build(pdf, output)
    loaded = ArticleBundle.model_validate_json((output / "bundle.json").read_text())
    assert bundle.parser == "pymupdf-fallback"
    assert loaded.metadata.title == "Synthetic ChemX article"
    assert loaded.metadata.page_count == 1
    assert "permeability" in loaded.pages[0].text
    assert loaded.pages[0].blocks
    assert (output / loaded.pages[0].render_path).is_file()

