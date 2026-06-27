---
name: chemx-parser
description: Route one ChemX article to its domain contract and extract complete, evidenced records.
---

# ChemX parser workflow

1. Read `bundle.json`, `layout.json`, `marker.md`, `marker.json`, `tables.json`, `ocr.json`, `ocsr.json`, `chemistry_candidates.json`, the selected domain `SKILL.md`, `domain.json`, and `output-schema.json`.
2. Treat tables as the primary row boundary. Reconstruct split headers and continuation rows before extraction.
3. If explicit tables are absent, extract rows from layout blocks, Marker Markdown, OCR text, figure captions, schemes, prose lists, and OCSR/RDKit chemistry candidates.
4. Inspect page renders and extracted figures when a value, chemical structure, superscript, or table cell is ambiguous.
5. Emit every record in the article. Do not summarize or silently deduplicate distinct experimental rows.
6. Do not return `records: []` when tables, OCR text, OCSR structures, chemistry candidates, compounds, targets, bacteria, metals, coformers, ratios, or photostability candidates are present.
7. Use the exact field names and scalar types from `domain.json`; never rename columns or invent schema fields.
8. Use `null` only when the domain rules require `NOT_DETECTED`; the deterministic normalizer maps null/NaN/ND to that token.
9. Preserve reported units and numeric precision. Do not convert units unless the domain contract explicitly requests it.
10. Attach page, source kind (`text`, `table`, `layout`, `marker`, `ocr`, `ocsr`, `figure`, `caption`, `metadata`), bounding box when known, and a short evidence excerpt for every field.
11. Never use gold, answer files, prior predictions, HuggingFace, or network lookups during extraction.
12. Return JSON only and satisfy `output-schema.json` exactly.
