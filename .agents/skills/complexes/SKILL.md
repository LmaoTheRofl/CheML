---
name: complexes
description: Extract Complexes records for ChemX using the exact local parquet gold columns.
---

# Complexes

Extract one record per ChemX table row for the current article DOI. Use the exact field names and scalar types from `domain.json`; do not rename, translate, merge, or omit columns.

Rules:
- Include every field listed below in every `values` object.
- Use `NOT_DETECTED`/null only when the value is genuinely absent from the article.
- Preserve source metadata columns (`doi`, `title`, `pdf`, `year`, etc.) when they are part of the contract.
- For numeric values, keep the value comparable to the parquet gold; decimal comma and dot are normalized by the evaluator.
- Attach page-level evidence for each extracted value whenever possible.
- Follow the ChemX Complexes schema: ligand canonical SMILES, metal, and thermodynamic target/logK rows are the core extraction target.
- When table extraction is empty, extract rows from prose, review lists, captions, schemes, OCR, and OCSR candidates. Complexes review articles often encode valid rows outside explicit tables.
- Do not return an empty `records` array if any metal/ligand/target candidates are present in `chemistry_candidates.json`, `ocr.json`, `layout.json`, or `marker.md`.
- Use RDKit-canonicalized SMILES candidates from `chemistry_candidates.json`/`ocsr.json` whenever they map to a compound row.

Fields:
- `pdf` (string)
- `doi` (string)
- `doi_sourse` (string)
- `supplementary` (integer)
- `title` (string)
- `publisher` (string)
- `year` (integer)
- `access` (integer)
- `compound_id` (string)
- `compound_name` (string)
- `SMILES` (string)
- `SMILES_type` (string)
- `metal` (string)
- `target` (string)
- `page_smiles` (integer)
- `origin_smiles` (string)
- `page_metal` (integer)
- `origin_metal` (string)
- `page_target_value` (number)
- `origin_target_value` (string)
