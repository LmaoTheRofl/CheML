---
name: chemx-parser
description: Route one ChemX article to its domain contract and extract complete, evidenced records.
---

# ChemX parser workflow

1. Read `bundle.json`, the selected domain `SKILL.md`, `domain.json`, and `output-schema.json`.
2. Treat tables as the primary row boundary. Reconstruct split headers and continuation rows before extraction.
3. Inspect page renders and extracted figures when a value, chemical structure, superscript, or table cell is ambiguous.
4. Emit every record in the article. Do not summarize or silently deduplicate distinct experimental rows.
5. Use `null` only when the domain rules require `NOT_DETECTED`; the deterministic normalizer maps null/NaN/ND to that token.
6. Preserve reported units and numeric precision. Do not convert units unless the domain contract explicitly requests it.
7. Attach page, source kind, bounding box when known, and a short evidence excerpt for every field.
8. Never use gold, answer files, prior predictions, HuggingFace, or network lookups during extraction.
9. Return JSON only and satisfy `output-schema.json` exactly.
