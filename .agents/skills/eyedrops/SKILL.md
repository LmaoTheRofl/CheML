---
name: eyedrops
description: Extract EyeDrops records for ChemX using the exact local parquet gold columns.
---

# EyeDrops

Extract one record per ChemX table row for the current article DOI. Use the exact field names and scalar types from `domain.json`; do not rename, translate, merge, or omit columns.

Rules:
- Include every field listed below in every `values` object.
- Use `NOT_DETECTED`/null only when the value is genuinely absent from the article.
- Preserve source metadata columns (`doi`, `title`, `pdf`, `year`, etc.) when they are part of the contract.
- For numeric values, keep the value comparable to the parquet gold; decimal comma and dot are normalized by the evaluator.
- Attach page-level evidence for each extracted value whenever possible.

Fields:
- `smiles` (string)
- `name` (string)
- `perm (cm/s)` (string)
- `logP` (string)
- `doi` (string)
- `PMID` (number)
- `title` (string)
- `publisher` (string)
- `year` (integer)
- `access` (integer)
- `page` (integer)
- `origin` (string)
