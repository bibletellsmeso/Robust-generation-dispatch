# Project Structure and Cleanup Map

## Current structure

The repository currently preserves multiple related implementation variants at the root. They use direct local imports, so moving Python modules before a tested refactor would break execution. The first cleanup pass therefore documents the variants and prevents new generated artefacts from being committed.

| Current path | Current role | Planned treatment |
|---|---|---|
| `RGD_Mc/` | CCG/MILP source variant and historical outputs | Retain while its entry point, parameters, and outputs are validated. |
| `RGD_Mc_QP/` | CCG/MIQP source variant and historical outputs | Retain while its entry point, parameters, and outputs are validated. |
| `RGD_bM/` | Big-M source variant with best/worst runs | Retain while method differences are documented. |
| `RGD_Mc_WT/` | Wind-related parameter material | Inventory before deciding whether it belongs with a variant or in shared configuration. |
| `data/` | Inputs and transformed data | Add provenance notes before renaming or distributing files. |
| `result/` | Existing tabular and binary results | Select a small final subset only after reproduction. |
| `**/export_*/` | Generated solver exports and figures | Preserve historical tracked files; exclude new unreviewed exports. |
| `**/__pycache__/`, `*.pyc` | Interpreter cache | Do not add in future commits. |
| `*.lp`, `*.ilp`, `*.log`, `*.pickle`, `*.pkl` | Solver intermediates and cached objects | Preserve history; do not add new copies without a reproducibility reason. |

## File-renaming policy

No existing source file is renamed in this initial branch. Names such as `CCG_SP_PMC copy.py` and historical mixed-case data filenames are candidates for cleanup, but a rename requires all of the following:

1. identify the importing and executing scripts;
2. assign a method-based name and record the old-to-new mapping;
3. update imports and paths together;
4. run at least a syntax check and a representative workflow;
5. retain the original file in a reviewable commit until the new path is accepted.

## Target layout

After the variants have been validated, the intended layout is:

```text
src/
  variants/
data/
  README.md
docs/
results/
  final/
  checkpoints/
tests/
```

This is a future migration target, not a claim that the current research code has already been refactored into a package.
