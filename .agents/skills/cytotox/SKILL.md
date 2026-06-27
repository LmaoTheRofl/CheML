---
name: cytotox
description: Extract Cytotox records for ChemX using the exact local parquet gold columns.
---

# Cytotox

Extract one record per ChemX table row for the current article DOI. Use the exact field names and scalar types from `domain.json`; do not rename, translate, merge, or omit columns.

Rules:
- Include every field listed below in every `values` object.
- Use `NOT_DETECTED`/null only when the value is genuinely absent from the article.
- Preserve source metadata columns (`doi`, `title`, `pdf`, `year`, etc.) when they are part of the contract.
- For numeric values, keep the value comparable to the parquet gold; decimal comma and dot are normalized by the evaluator.
- Attach page-level evidence for each extracted value whenever possible.

Fields:
- `sn` (integer)
- `material` (string)
- `shape` (string)
- `coat_functional_group` (string)
- `synthesis_method` (string)
- `surface_charge` (string)
- `size_in_medium_nm` (number)
- `zeta_in_medium_mv` (number)
- `no_of_cells_cells_well` (number)
- `human_animal` (string)
- `cell_source` (string)
- `cell_tissue` (string)
- `cell_morphology` (string)
- `cell_age` (string)
- `time_hr` (integer)
- `concentration` (number)
- `test` (string)
- `test_indicator` (string)
- `viability_%` (number)
- `doi` (string)
- `article_list` (integer)
- `core_nm` (number)
- `hydrodynamic_nm` (number)
- `potential_mv` (number)
- `cell_type` (string)
- `journal_name` (string)
- `publisher` (string)
- `year` (integer)
- `title` (string)
- `journal_is_oa` (boolean)
- `is_oa` (string)
- `oa_status` (string)
- `pdf` (string)
- `access` (integer)
