---
name: nanozymes
description: Extract Nanozymes records for ChemX using the exact local parquet gold columns.
---

# Nanozymes

Extract one record per ChemX table row for the current article DOI. Use the exact field names and scalar types from `domain.json`; do not rename, translate, merge, or omit columns.

Rules:
- Include every field listed below in every `values` object.
- Use `NOT_DETECTED`/null only when the value is genuinely absent from the article.
- Preserve source metadata columns (`doi`, `title`, `pdf`, `year`, etc.) when they are part of the contract.
- For numeric values, keep the value comparable to the parquet gold; decimal comma and dot are normalized by the evaluator.
- Attach page-level evidence for each extracted value whenever possible.

Fields:
- `formula` (string)
- `activity` (string)
- `syngony` (string)
- `length` (string)
- `width` (string)
- `depth` (string)
- `surface` (string)
- `km_value` (number)
- `km_unit` (string)
- `vmax_value` (number)
- `vmax_unit` (string)
- `target_source` (string)
- `reaction_type` (string)
- `c_min` (number)
- `c_max` (number)
- `c_const` (number)
- `c_const_unit` (string)
- `ccat_value` (number)
- `ccat_unit` (string)
- `ph` (number)
- `temperature` (number)
- `doi` (string)
- `pdf` (string)
- `access` (integer)
- `title` (string)
- `journal` (string)
- `year` (integer)
