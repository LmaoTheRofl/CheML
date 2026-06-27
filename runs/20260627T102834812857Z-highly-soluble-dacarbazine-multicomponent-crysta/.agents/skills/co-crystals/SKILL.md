---
name: co-crystals
description: Extract Co-crystals records for ChemX using the exact local parquet gold columns.
---

# Co-crystals

Extract one record per ChemX table row for the current article DOI. Use the exact field names and scalar types from `domain.json`; do not rename, translate, merge, or omit columns.

Rules:
- Include every field listed below in every `values` object.
- Use `NOT_DETECTED`/null only when the value is genuinely absent from the article.
- Preserve source metadata columns (`doi`, `title`, `pdf`, `year`, etc.) when they are part of the contract.
- For numeric values, keep the value comparable to the parquet gold; decimal comma and dot are normalized by the evaluator.
- Attach page-level evidence for each extracted value whenever possible.

Fields:
- `pdf` (string)
- `doi` (string)
- `supplementary` (integer)
- `authors` (string)
- `title` (string)
- `journal` (string)
- `year` (integer)
- `page` (number)
- `access` (integer)
- `name_cocrystal` (string)
- `name_cocrystal_type_file` (string)
- `name_cocrystal_page` (string)
- `name_cocrystal_origin` (integer)
- `ratio_cocrystal` (string)
- `ratio_cocrystal_page` (string)
- `ratio_cocrystal_page.1` (string)
- `ratio_cocrystal_origin` (number)
- `name_drug` (string)
- `name_drug_type_file` (string)
- `name_drug_origin` (string)
- `name_drug_page` (integer)
- `SMILES_drug` (string)
- `SMILES_drug_type_file` (string)
- `SMILES_drug_origin` (string)
- `SMILES_drug_page` (number)
- `name_coformer` (string)
- `name_coformer_type file` (string)
- `name_coformer_origin` (string)
- `name_coformer_page` (number)
- `SMILES_coformer` (string)
- `SMILES_coformer_type file` (string)
- `SMILES_coformer_origin` (string)
- `SMILES_coformer_page` (number)
- `photostability_change` (string)
- `photostability_change_type_file` (string)
- `photostability_change_origin` (string)
- `photostability_change_page` (number)
