"""Repository-relative paths for the canonical RGD_Mc workflow."""

from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = MODULE_DIR.parent
DATA_DIR = REPOSITORY_ROOT / "data"
RESULT_DIR = REPOSITORY_ROOT / "result"
RUNTIME_DIR = MODULE_DIR / "runtime"
