# Robust Generation Dispatch

Research code and experiment records for robust generation dispatch in renewable-rich microgrids. The repository contains several implementation variants of a two-stage adaptive robust optimization workflow, including column-and-constraint generation (CCG), MILP/MIQP formulations, photovoltaic-data processing, and preserved experiment outputs.

## Status

This repository is being documented and cleaned in a review branch. Historical files are preserved while the source, data, and generated artefacts are mapped. The current code is research software, not a packaged library.

## Repository map

| Path | Role |
|---|---|
| `RGD_Mc/` | Main CCG/MILP implementation variant, including `CCG_algo.py`. |
| `RGD_Mc_QP/` | CCG/MIQP implementation variant, including `CCG_algo_QP.py`. |
| `RGD_bM/` | Big-M implementation variant with separate best- and worst-case CCG scripts. |
| `RGD_Mc_WT/` | Wind-turbine-related parameter material; its run path still needs documentation. |
| `data/` | Input weather, PV, load, wind, and price data. See [data provenance](docs/DATA_PROVENANCE.md). |
| `result/` | Preserved summary outputs from prior experiments. |
| `PV_generation.py` | Standalone photovoltaic data-processing script; it currently contains a machine-specific input path. |

The names `Mc`, `Mc_QP`, `bM`, and `WT` are historical variant labels. Their scientific expansion and recommended entry point will be confirmed before any source-file rename.

## Requirements

- Python 3.8 or later
- A working [Gurobi](https://www.gurobi.com/) installation and license for optimization runs
- The Python packages in `requirements.txt`

Create an isolated environment and install the Python dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`gurobipy` also requires Gurobi to be installed and licensed; installing the Python package alone is not enough.

## Running an existing variant

The implementation uses imports relative to each variant directory. Run a selected script from inside that directory, after reviewing its `Params.py` configuration and input paths. For example:

```powershell
Set-Location RGD_Mc
python CCG_algo.py
```

For the quadratic-programming variant, use `RGD_Mc_QP\CCG_algo_QP.py`. The Big-M variant has separate `CCG_algo_best.py` and `CCG_algo_worst.py` entry scripts.

Before a full run, confirm that the input data referenced by the selected variant is available and that the chosen Gurobi license is active. A reproducible minimal example will be added after the current variants and parameters are validated.

### Current RGD_Mc run status

The `RGD_Mc` entry point now uses repository-relative paths for its primary inputs and runtime output. It reaches the first CCG dual solve on a fresh clone with the documented dependencies and a Gurobi license. The currently committed 2025-01-15 configuration then reports an infeasible dual subproblem; this is a model/scenario validation question, not a missing-path or missing-package error. The generated diagnostic stays under ignored `RGD_Mc/runtime/`.

## Data and results

Input files are retained in `data/`. Their origins, permissions, and transformation status must be recorded before redistribution or major restructuring. Existing solver models, caches, pickles, and repeated plots are historical experiment artefacts. New disposable output is excluded by `.gitignore`; selected reproducible figures and tables should eventually be curated into a documented `results/final/` location. See the [legacy artifact inventory](docs/LEGACY-ARTIFACT-INVENTORY.md) before removing or relocating retained files.

## Reproducibility and citation

Please open an issue before relying on this code for a new study. The project does not yet provide a frozen release, a verified end-to-end reproduction script, or a formal citation record. These will be added after the cleanup branch has been reviewed.

## Cleanup policy

See [the project structure note](docs/PROJECT_STRUCTURE.md) and the account-wide [public repository standard](https://github.com/bibletellsmeso/research-hub/blob/main/docs/PUBLIC-REPOSITORY-STANDARD.md). No historical files are deleted or renamed by this documentation-only change.
