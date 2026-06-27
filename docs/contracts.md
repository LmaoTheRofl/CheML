# Контракты данных

## ArticleBundle

`bundle.json` содержит `metadata`, `pages`, `blocks`, `tables`, `figures`, пути к renders/assets и координаты в PDF points. Все пути относительны к run workspace. `schema_version` сейчас равен `1.0`.

## Prediction

Prediction содержит `domain` и массив `records`. Каждый record имеет `values` по всем полям domain contract и `evidence` с page, kind, bbox, excerpt и asset path. Неизвестные поля запрещены generated JSON Schema.

После успешного inference run workspace содержит:

- `prediction.json` — structured output backend;
- `prediction.csv` — те же `values` в плоской таблице для ручного сравнения;
  имена и порядок столбцов совпадают с `domain.json` и локальным
  parquet/CSV gold в `datasets`;
- `bundle.json`, `manifest.json`, `output-schema.json` и `assets/`.

## Нормализация и оценка

- missing/NaN/ND → `NOT_DETECTED`;
- decimal comma → point;
- числовые trailing zeros удаляются без float-rounding;
- SMILES → canonical RDKit SMILES, если RDKit установлен;
- precision/recall/F1 считаются по точному multiset intersection каждого столбца.
- `macro_f1` статьи — невзвешенное арифметическое среднее F1 всех её столбцов.

Gold запрещён в inference workspace и читается только evaluator после завершённого inference. Default gold source — локальный parquet в `datasets`, отфильтрованный по DOI текущей статьи; если DOI не совпал, evaluator пробует найти строки по столбцу `pdf`, затем по нормализованному `title`. Метрики записываются в `evaluation.json` и `evaluation_metrics.csv`.

Перед чтением parquet evaluator требует точного совпадения имён, порядка и
скалярных типов Arrow с domain contract: `string`, integer → `integer`,
floating/decimal → `number`, boolean → `boolean`. `chemx audit-schemas datasets`
записывает построчный отчёт `runs/parquet_schema_audit.csv`.

`chemx evaluate-batch runs` оценивает новейший завершённый run каждого
уникального `source_pdf` и создаёт `runs/article_macro_f1.csv` со столбцами
`run_id`, `source_pdf`, `domain`, DOI, количеством predicted/gold строк,
числом полей, `macro_f1`, флагами полного F1 и совпадения схемы, а также путями
к gold и run.
