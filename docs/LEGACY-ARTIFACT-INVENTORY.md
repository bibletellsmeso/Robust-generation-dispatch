# Legacy artifact inventory

This repository retains historical optimization runs alongside research code. The files below are not a curated, versioned result release; they are compatibility material for legacy scripts and a record of earlier experiments.

## Counted generated artifacts

| Type | Files | Approximate size | Current treatment |
|---|---:|---:|---|
| Solver model exports (`.lp`) | 16 | 5.82 MB | Generated diagnostics; ignored for future runs. |
| Infeasibility models (`.ilp`) | 3 | 46 KB | Generated diagnostics; ignored for future runs. |
| Serialized run state (`.pickle`) | 68 | 64 KB | Legacy scripts may read these during post-processing. |
| Local statistics (`.pkl`) | 4 | 21 KB | At least one variant loads them directly. |

The counts above were recorded on 2026-09-01. The repository `.gitignore` prevents new files of these types from being added accidentally, but already tracked files remain until their dependent code is refactored and validated.

## Why they are retained for now

- `RGD_Mc/Data_read.py` loads local PV and ramping statistics from `.pkl` files.
- Multiple CCG scripts read prior `.pickle` state or write solver diagnostics into historical `export_*` paths.
- Several scripts still contain machine-specific paths, so blindly relocating files would make the public snapshot less runnable, not more reproducible.

## Runtime smoke-test result

On 2026-09-01, a local Python environment installed the declared dependencies, `SciencePlots`, and a working Gurobi academic license. A minimal Gurobi optimization solved successfully. The main `RGD_Mc/CCG_algo.py` entry point then stopped before optimization because `RGD_Mc/PV_model.py` requests a legacy absolute weather-data path that does not exist on a fresh clone. This confirms that path normalization—not solver availability—is the first repair required for an end-to-end run.

`SciencePlots` is listed in `requirements.txt` because the CCG and plotting scripts explicitly select the `science` Matplotlib style.

## Safe removal plan

1. Replace the 169 machine-specific path references with repository-relative configuration, beginning with the main `RGD_Mc` variant.
2. Make each variant regenerate any required statistics and runtime state.
3. Run a small licensed-Gurobi validation for the chosen canonical variant.
4. Retain only documented figures and tables in a clear final-results location.
5. Remove the now-unneeded tracked artifacts in one reviewable commit.

## Public data caution

`data/KPX_PV.csv` and `data/KPX_WT.csv` are large input datasets. Their provenance and redistribution permission must be confirmed before moving, mirroring, or presenting them as a general-purpose public dataset.
