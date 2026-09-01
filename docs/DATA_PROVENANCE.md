# Data Provenance Register

The data directory contains weather, photovoltaic, load, wind, price, and scheduling input files from multiple experiments. This initial register records their status without changing or republishing the files.

| Group | Examples | Current status | Required follow-up |
|---|---|---|---|
| Hawaii weather | `25109_21.29_-157.86_*.csv`, `202501*_Hawaii_weather.*` | Historical input and transformed weather files | Record original provider, retrieval date, license, and transformation script. |
| Korean power-system series | `KPX_Load.*`, `KPX_PV.csv`, `KPX_WT.csv`, `smp_land_*.xls` | Historical input data | Confirm source, redistribution permission, units, and time zone. |
| Scheduling series | `Load_for_scheduling.txt`, `PV_for_scheduling.txt`, `WT_for_scheduling.txt` | Derived experiment inputs | Identify the generating script and parent input version. |
| Result summaries | `result/*.csv`, `result/*.xlsx`, `result/*.pickle` | Historical outputs/caches | Select only reproducible final artefacts for future publication. |

Do not add credentials, licensed raw datasets, or unpublished research data to this public repository. When an input cannot be redistributed, document how an authorized user can obtain it instead.
