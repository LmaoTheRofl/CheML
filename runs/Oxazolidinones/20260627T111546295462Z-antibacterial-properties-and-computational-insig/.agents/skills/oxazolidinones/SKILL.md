---
name: oxazolidinones
description: Extract Oxazolidinones records for ChemX using the exact local parquet gold columns.
---

# Oxazolidinones

Extract one record per ChemX table row for the current article DOI. Use the exact field names and scalar types from `domain.json`; do not rename, translate, merge, or omit columns.

Rules:
- Include every field listed below in every `values` object.
- Use `NOT_DETECTED`/null only when the value is genuinely absent from the article.
- Preserve source metadata columns (`doi`, `title`, `pdf`, `year`, etc.) when they are part of the contract.
- For numeric values, keep the value comparable to the parquet gold; decimal comma and dot are normalized by the evaluator.
- Attach page-level evidence for each extracted value whenever possible.

Fields:
- `pdf` (string)
- `smiles` (string)
- `doi` (string)
- `title` (string)
- `publisher` (string)
- `year` (integer)
- `access` (integer)
- `compound_id` (string)
- `target_type` (string)
- `target_relation` (string)
- `target_value` (string)
- `target_units` (string)
- `bacteria` (string)
- `bacteria_name_unified` (string)
- `bacteria_info` (string)
- `page_bacteria` (integer)
- `origin_bacteria` (string)
- `section_bacteria` (string)
- `subsection_bacteria` (string)
- `page_target` (integer)
- `origin_target` (string)
- `section_target` (string)
- `subsection_target` (number)
- `column_prop` (string)
- `line_prop` (number)
- `page_scaffold` (integer)
- `origin_scaffold` (string)
- `section_scaffold` (string)
- `subsection_scaffold` (number)
- `page_residue` (number)
- `origin_residue` (string)
- `section_residue` (string)
