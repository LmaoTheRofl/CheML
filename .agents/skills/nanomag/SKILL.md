---
name: nanomag
description: Extract Nanomag records for ChemX using the exact local parquet gold columns.
---

# Nanomag

Extract one record per ChemX table row for the current article DOI. Use the exact field names and scalar types from `domain.json`; do not rename, translate, merge, or omit columns.

Rules:
- Include every field listed below in every `values` object.
- Use `NOT_DETECTED`/null only when the value is genuinely absent from the article.
- Preserve source metadata columns (`doi`, `title`, `pdf`, `year`, etc.) when they are part of the contract.
- For numeric values, keep the value comparable to the parquet gold; decimal comma and dot are normalized by the evaluator.
- Attach page-level evidence for each extracted value whenever possible.

Fields:
- `name` (string)
- `np_shell_2` (string)
- `np_hydro_size` (number)
- `xrd_scherrer_size` (number)
- `zfc_h_meas` (number)
- `htherm_sar` (number)
- `mri_r1` (number)
- `mri_r2` (number)
- `emic_size` (number)
- `instrument` (string)
- `core_shell_formula` (string)
- `np_core` (string)
- `np_shell` (string)
- `space_group_core` (string)
- `space_group_shell` (string)
- `squid_h_max` (number)
- `fc_field_T` (string)
- `squid_temperature` (string)
- `squid_sat_mag` (string)
- `coercivity` (string)
- `squid_rem_mag` (string)
- `exchange_bias_shift_Oe` (string)
- `vertical_loop_shift_M_vsl_emu_g` (string)
- `hc_kOe` (number)
- `doi` (string)
- `pdf` (string)
- `supp` (string)
- `journal` (string)
- `publisher` (string)
- `year` (integer)
- `title` (string)
- `access` (number)
- `verification required` (number)
- `verified_by` (number)
- `verification_date` (number)
- `has_mistake_in_matadata` (number)
- `comment` (number)
- `article_name_folder` (string)
- `supp_info_name_folder` (string)
