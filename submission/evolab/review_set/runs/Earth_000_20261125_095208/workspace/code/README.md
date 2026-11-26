# Code: GlaMBIE-style reconciliation of glacier mass change 2000–2023

Run everything from the workspace root: `bash code/run_all.sh` (≈2–3 min).

| order | script | purpose | main outputs |
|---|---|---|---|
| 1 | `inventory.py` | inventory of the 257 input files (region, source, unit, span, step, errors) | `outputs/inventory.csv` |
| 2 | `calendar_weights.py` | calendar-conversion weights: from the monthly glaciological sine-wave product (used) and implied by official HY/CY pairs (diagnostic only) | `outputs/calendar_weights.csv` |
| 3 | `run_consensus.py` | runs the main configuration and 13 sensitivity configurations, writes regional HY/CY tables and global series, and computes reproduction metrics against the official results | `outputs/configs/<cfg>/`, `outputs/config_metrics.csv` |
| 4 | `analysis.py` | global and regional statistics, ice-sheet comparison, relative loss, method consistency, z-test, hybrid validation, sensitivity, projections | `outputs/summary_stats.json`, `outputs/*.csv` |
| 5 | `make_figures.py` | all figures | `report/images/*.png` |
| 6 | `build_report.py` | writes `report/report.md`, taking all numbers from the CSV/JSON outputs | `report/report.md` |

`glambie_lib.py` contains the method: homogenisation, annualisation, group combination, error model and calendar conversion. `compare_hy.py` compares hydrological-year series with the official files.
The official results (`data/glambie/results`) are used only for (i) the auxiliary area model A(t), (ii) validation, and (iii) choosing the main configuration among the predefined variants.
