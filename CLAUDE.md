# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

French real estate mutation analysis (DVF - Demandes de Valeurs Foncières). Loads official pipe-separated DVF data from data.gouv.fr (2020-2025), cleans/aggregates it, and trains ML models (RandomForest) to predict property values and transaction duration. Current focus: house-only mutations, with specific interest in Ensues.

## Commands

```bash
uv sync                          # Install dependencies
uv sync --extra dev --extra geo  # Install with dev (pytest, ruff) and geo (geopandas) extras
uv run pytest                    # Run all tests
uv run pytest tests/test_load.py # Run a single test file
uv run ruff check .              # Lint
uv run ruff format .             # Format
uv run jupyter lab               # Launch JupyterLab for notebooks
uv run python scripts/download_dvf_sample.py  # Download DVF data
```

## Architecture

```
src/dvf/          # Reusable Python package
  load.py         # load_dvf_raw() (pipe-sep), load_dvf_csv(), load_dvf_plus() (enriched DVF+)
  analyze.py      # summarize_mutations(), price_stats()

notebooks/        # Sequential analysis pipeline (01-06)
  01: Ensues house exploration
  02: France-wide house mutations
  03: Data cleaning & deduplication
  04: Train/test split + RandomForest regression
  05: Survival analysis
  06: EDA on cleaned France dataset

scripts/          # Data acquisition utilities
config/defaults.yaml  # Paths, encoding (utf-8), separator ("|")
```

**Data pipeline**: `data/raw/*.txt` → load functions → cleaning (notebook 03) → `data/processed/*.csv` → modeling → `data/models/*.pkl` + `data/results/`

## Key Conventions

- **Package manager**: UV (not pip). Always use `uv run` or `uv sync`.
- **Python**: 3.11+
- **Raw DVF files**: pipe-separated (`|`), not comma-separated.
- **Line length**: 100 (ruff config in pyproject.toml)
- **Build backend**: hatchling
- **Source layout**: `src/dvf/` (configured as pythonpath in pytest)
- **Comments**: Use comments sparingly. Only add comments where the logic isn't self-evident.
