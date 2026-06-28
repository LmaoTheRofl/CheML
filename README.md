# ChemX Article Parser

Воспроизводимый Python 3.11 pipeline для построения `ArticleBundle`, извлечения ChemX-таблиц через один изолированный `codex exec` на статью и точной оценки по multiset-метрике.

## Быстрый старт

```bash
UV_CACHE_DIR=runs/tools/cache/uv UV_PYTHON_INSTALL_DIR=runs/tools/python uv python install 3.11
UV_CACHE_DIR=runs/tools/cache/uv UV_PYTHON_INSTALL_DIR=runs/tools/python uv sync --extra dev
UV_CACHE_DIR=runs/tools/cache/uv UV_PYTHON_INSTALL_DIR=runs/tools/python uv sync --extra ui
uv run chemx inspect datasets/NANOMATERIALS/SelTox/d3ra07733k.pdf
uv run chemx bundle datasets/NANOMATERIALS/SelTox/d3ra07733k.pdf --output-dir runs/seltox-bundle --no-marker
uv run chemx parse datasets/NANOMATERIALS/SelTox/d3ra07733k.pdf --domain auto
uv run chemx ui
uv run chemx batch datasets/
uv run chemx resume runs/<failed-run-id>
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

Если внешний Codex/Ollama backend недоступен после завершённого preprocessing,
`chemx resume runs/<failed-run-id>` повторяет только inference и reviewer,
переиспользуя сохранённые bundle/Marker/OCR/OCSR artifacts без повторного
парсинга PDF.
Если reviewer нужно пропустить для экономии Codex quota, используйте
`--no-reviewer` с `parse`, `batch` или `resume`.

Production `chemx parse` and `chemx batch` use Codex by default and require the
full parser stack: PyMuPDF, `pymupdf_layout`, Marker, local OCR, MolScribe,
RDKit, Codex extraction, and Codex reviewer. The project automatically discovers
the installed OCR data under `runs/tools/tesseract` and MolScribe runtime/weights
under `runs/tools`; `CHEMX_OCR_COMMAND` and `CHEMX_MOLSCRIBE_COMMAND` remain
explicit overrides. Missing tools fail fast through `chemx doctor-tools` or at
parse startup; the pipeline does not silently fall back to a weak mode.

Codex runner использует `gpt-5.5`, `model_reasoning_effort="xhigh"`, `--ephemeral`, `--sandbox workspace-write` и domain-specific `--output-schema`. Для локального backend контракт тот же:

```bash
OLLAMA_MODELS="$PWD/.ollama/models" ~/.local/bin/ollama pull lukaspetrik/gemma3-tools:27b
scripts/ollama_serve.sh
uv run chemx parse article.pdf --domain seltox --backend ollama
```

Локальный backend использует `lukaspetrik/gemma3-tools:27b` (27.4B, Q4_K_M)
через совместимый с `codex exec --oss` adapter на `127.0.0.1:11434`. Launcher
поднимает Ollama upstream на `127.0.0.1:11435`, если он ещё не запущен, и затем adapter.
Adapter отключает конфликтующую schema grammar только во время tool calls,
оставляет доступными `exec_command`, `write_stdin` и `view_image`, а итоговый
`prediction.json` по-прежнему проверяется Pydantic-контрактом pipeline. Адреса
можно переопределить через `CHEMX_OLLAMA_ADAPTER_URL` и `OLLAMA_UPSTREAM_URL`.
Команды `parse` и `batch` используют Codex по умолчанию. Локальный backend
включается только явно через `--backend ollama`.

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

Streamlit UI открывает одностраничное приложение для загрузки PDF, ручного
выбора домена и запуска `codex` или `ollama` backend. В таблице результата
показываются только поля, которые участвуют в оценке: список берётся из первого
столбца соответствующего `metrics/*_from_single_agent.csv`. Домены без такого
baseline-файла в UI не показываются. Во время запуска UI показывает краткий
stage/log из текущего run workspace и обновляет его автоматически. Для отмены
используйте кнопку `Stop extraction` внутри приложения: она завершает весь
process group активного запуска, включая Marker/Codex/Ollama-потомков.
Встроенные кнопки Streamlit `Rerun`/`Stop` относятся к перезапуску UI-скрипта и
не являются штатным способом запуска или остановки ChemX extraction.

Domain contracts находятся в `.agents/skills`; runtime JSON Schema генерируется из `domain.json`, поэтому prompt, validator и backend используют единый список полей. Архитектура и форматы описаны в [docs/architecture.md](docs/architecture.md) и [docs/contracts.md](docs/contracts.md).

Команды подготовки пакетов и локальных моделей:

```bash
UV_CACHE_DIR=runs/tools/cache/uv UV_PYTHON_INSTALL_DIR=runs/tools/python uv sync --extra dev --extra ui --extra gold
uv run python scripts/download_datalab_cache.py
OLLAMA_MODELS="$PWD/.ollama/models" ~/.local/bin/ollama pull lukaspetrik/gemma3-tools:27b
uv run chemx doctor-tools
```

`scripts/download_datalab_cache.py` заполняет Datalab/Marker cache из
`https://models.datalab.to`, но не скачивает отсутствующий `model.safetensors`:
этот вес должен уже лежать в ожидаемом cache path, иначе скрипт завершится
ошибкой. Marker, RDKit и `pymupdf_layout` являются обязательными
runtime-зависимостями parser. MolScribe использует обнаруженное project-local
окружение `runs/tools/molscribe-py39` и установленный `.pth`; OCR использует
project-local `runs/tools/tesseract/.../eng.traineddata`. Пути можно
переопределить через `CHEMX_MOLSCRIBE_COMMAND` и `CHEMX_OCR_COMMAND`.

## Проверки

```bash
uv run pytest
uv run ruff check .
```

Целевой `F1=1.0` проверяется только после реального inference и локального parquet-gold.
Репозиторные тесты не скачивают gold и не выполняют платный inference.
