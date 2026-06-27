---
name: synergy
description: Extract Synergy records for ChemX using the exact local parquet gold columns.
---

# Synergy

Extract one record per ChemX table row for the current article DOI. Use the exact field names and scalar types from `domain.json`; do not rename, translate, merge, or omit columns.

Rules:
- Include every field listed below in every `values` object.
- Use `NOT_DETECTED`/null only when the value is genuinely absent from the article.
- Preserve source metadata columns (`doi`, `title`, `pdf`, `year`, etc.) when they are part of the contract.
- For numeric values, keep the value comparable to the parquet gold; decimal comma and dot are normalized by the evaluator.
- Attach page-level evidence for each extracted value whenever possible.

Fields:
- `sn` (integer)
- `NP` (string)
- `bacteria` (string)
- `strain` (string)
- `NP_synthesis` (string)
- `drug` (string)
- `drug_dose_µg_disk` (number)
- `NP_concentration_µg_ml` (number)
- `NP_size_min_nm` (string)
- `NP_size_max_nm` (number)
- `NP_size_avg_nm` (number)
- `shape` (string)
- `method` (string)
- `ZOI_drug_mm_or_MIC _µg_ml` (number)
- `error_ZOI_drug_mm_or_MIC_µg_ml` (number)
- `ZOI_NP_mm_or_MIC_np_µg_ml` (number)
- `error_ZOI_NP_mm_or_MIC_np_µg_ml` (number)
- `ZOI_drug_NP_mm_or_MIC_drug_NP_µg_ml` (number)
- `error_ZOI_drug_NP_mm_or_MIC_drug_NP_µg_ml` (number)
- `fold_increase_in_antibacterial_activity` (number)
- `zeta_potential_mV` (string)
- `MDR` (string)
- `FIC` (number)
- `effect` (string)
- `reference` (string)
- `doi` (string)
- `article_list` (integer)
- `time_hr` (number)
- `coating_with_antimicrobial_peptide_polymers` (string)
- `combined_MIC` (number)
- `peptide_MIC` (number)
- `viability_%` (number)
- `viability_error` (number)
- `journal_name` (string)
- `publisher` (string)
- `year` (integer)
- `title` (string)
- `journal_is_oa` (boolean)
- `is_oa` (boolean)
- `oa_status` (string)
- `pdf` (string)
- `access` (integer)
