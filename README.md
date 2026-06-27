# ChemX Article Parser

Воспроизводимый Python 3.11 pipeline для построения `ArticleBundle`, извлечения ChemX-таблиц через один изолированный `codex exec` на статью и точной оценки по multiset-метрике.

## Быстрый старт

```bash
UV_CACHE_DIR=/tmp/uv-cache UV_PYTHON_INSTALL_DIR=/tmp/uv-python uv python install 3.11
UV_CACHE_DIR=/tmp/uv-cache UV_PYTHON_INSTALL_DIR=/tmp/uv-python uv sync --extra dev
uv run chemx inspect datasets/NANOMATERIALS/SelTox/d3ra07733k.pdf
uv run chemx bundle datasets/NANOMATERIALS/SelTox/d3ra07733k.pdf --output-dir runs/seltox-bundle --no-marker
uv run chemx parse datasets/NANOMATERIALS/SelTox/d3ra07733k.pdf --domain auto --backend codex
uv run chemx batch datasets/
uv run chemx doctor-tools
uv run chemx inspect-run runs/<run-id>
```

Run output directory must be persistent: `chemx parse` and `chemx batch` reject
`--runs-dir` paths inside `/tmp`. Each completed inference run writes both
`prediction.json` and `prediction.csv` under `runs/<run-id>/`. Columns in
`prediction.csv` follow the domain contract order, matching the local
parquet/CSV gold schema in `datasets`. After `chemx evaluate`, the same run
also contains `reference.csv`: the filtered parquet rows for that article in
the same column order as `prediction.csv`.

Production `chemx parse --backend codex` requires the full parser stack:
PyMuPDF, `pymupdf_layout`, Marker, local OCR, MolScribe via
`CHEMX_MOLSCRIBE_COMMAND`, RDKit, Codex extraction, and Codex reviewer. Missing
tools fail fast through `chemx doctor-tools` or at parse startup; the pipeline no
longer silently falls back to a weak extraction mode.

Codex runner использует `gpt-5.5`, `model_reasoning_effort="xhigh"`, `--ephemeral`, `--sandbox workspace-write` и domain-specific `--output-schema`. Для локального backend контракт тот же:

```bash
uv run chemx parse article.pdf --domain seltox --backend ollama
```

## Оценка

Gold не копируется в inference workspace и может загружаться только после состояния `inference_complete`:

```bash
uv sync --extra gold
uv run chemx evaluate runs/<run-id>
uv run chemx evaluate runs/<run-id> --gold /path/to/gold.json
uv run chemx evaluate-batch runs
uv run chemx audit-schemas datasets
```

По умолчанию evaluator использует скачанные локальные parquet-файлы в
`datasets/**/train-*.parquet`, выбирает parquet рядом с исходным PDF и фильтрует
строки по DOI текущей статьи, если DOI не совпал — по столбцу `pdf`, а если
и он не сопоставился — по нормализованному `title`.
`RiddarsCorp/test_chemx/exp_final.xlsx` не
используется: это Q&A workbook, а не табличный ChemX gold. Явный `--gold`
поддерживает JSON/CSV/XLSX/parquet, но Q&A-таблицы отклоняются.

`NaN`, пустые значения и `ND` нормализуются в `NOT_DETECTED`; десятичная запятая заменяется точкой; SMILES канонизируются RDKit при установленном extra `chemistry`. Для каждого столбца считаются precision/recall/F1 по точному multiset-пересечению.
Результаты оценки пишутся в `evaluation.json`, `evaluation_metrics.csv` и
`reference.csv`.
`macro_f1` статьи — арифметическое среднее F1 всех её полей. Команда
`evaluate-batch` выбирает новейший завершённый run для каждого исходного PDF и
записывает `runs/article_macro_f1.csv` (одна строка на статью). Команда
`audit-schemas` проверяет имена, порядок и Arrow-типы полей всех 10 локальных
parquet относительно `domain.json`; отчёт сохраняется в
`runs/parquet_schema_audit.csv`.

## Optional-компоненты

```bash
uv sync --extra ui
uv run chemx ui
```

Domain contracts находятся в `.agents/skills`; runtime JSON Schema генерируется из `domain.json`, поэтому prompt, validator и backend используют единый список полей. Архитектура и форматы описаны в [docs/architecture.md](docs/architecture.md) и [docs/contracts.md](docs/contracts.md).

Marker, RDKit и `pymupdf_layout` являются обязательными runtime-зависимостями parser. MolScribe должен быть установлен в отдельном окружении и доступен через `CHEMX_MOLSCRIBE_COMMAND`; OCR задаётся через `CHEMX_OCR_COMMAND` или системный `tesseract`.

## Проверки

```bash
uv run pytest
uv run ruff check .
```

Целевой `F1=1.0` проверяется только после реального inference и локального parquet-gold.
Репозиторные тесты не скачивают gold и не выполняют платный inference.
