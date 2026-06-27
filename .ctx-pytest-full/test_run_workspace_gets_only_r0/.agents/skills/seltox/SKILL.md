---
name: seltox
description: Extract SelTox records for ChemX using the exact local parquet gold columns.
---

# SelTox

Extract one record per ChemX table row for the current article DOI. Use the exact field names and scalar types from `domain.json`; do not rename, translate, merge, or omit columns.

Rules:
- Include every field listed below in every `values` object.
- Use `NOT_DETECTED`/null only when the value is genuinely absent from the article.
- Preserve source metadata columns (`doi`, `title`, `pdf`, `year`, etc.) when they are part of the contract.
- For numeric values, keep the value comparable to the parquet gold; decimal comma and dot are normalized by the evaluator.
- Attach page-level evidence for each extracted value whenever possible.

Fields:
- `sn` (integer)
- `np` (string)
- `coating` (integer)
- `bacteria` (string)
- `mdr` (integer)
- `strain` (string)
- `np_synthesis` (string)
- `method` (string)
- `mic_np_µg_ml` (string)
- `concentration` (number)
- `zoi_np_mm` (number)
- `np_size_min_nm` (number)
- `np_size_max_nm` (number)
- `np_size_avg_nm` (number)
- `shape` (string)
- `time_set_hours` (number)
- `zeta_potential_mV` (number)
- `solvent_for_extract` (string)
- `temperature_for_extract_C` (number)
- `duration_preparing_extract_min` (number)
- `precursor_of_np` (string)
- `concentration_of_precursor_mM` (number)
- `hydrodynamic_diameter_nm` (number)
- `ph_during_synthesis` (number)
- `reference` (string)
- `doi` (string)
- `article_list` (integer)
- `journal_name` (string)
- `publisher` (string)
- `year` (integer)
- `title` (string)
- `journal_is_oa` (boolean)
- `is_oa` (boolean)
- `oa_status` (string)
- `pdf` (string)
- `access` (integer)
