# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

French real estate mutation analysis (DVF - Demandes de Valeurs Foncières). Loads official pipe-separated DVF data from data.gouv.fr (2020-2025), cleans/aggregates it. Current focus: **the dataset of Marseille houses (Maison) sold on a target date** (default 15/09/2025), each enriched with the JSON prior-sale history of its cadastral parcel — the processed CSV is consumed by a separate downstream project, **vision360immeuble**. A Streamlit app remains for ad-hoc structured querying of the data.

## Commands

```bash
uv sync                          # Install dependencies
uv sync --extra dev --extra geo  # Install with dev (pytest, ruff) and geo (geopandas) extras
uv run pytest                    # Run all tests
uv run pytest tests/test_load.py # Run a single test file
uv run ruff check .              # Lint
uv run ruff format .             # Format
uv run python scripts/download_dvf_sample.py            # Download DVF data
uv run python scripts/build_marseille_houses_sold_on_date.py             # Build houses-sold dataset (default 15/09/2025)
uv run python scripts/build_marseille_houses_sold_on_date.py --date 15/09/2025  # Same, explicit date
uv run streamlit run app.py      # Launch chat interface for querying DVF data
```

## Architecture

```
src/dvf/          # Reusable Python package
  load.py         # load_dvf_raw() (pipe-sep), load_dvf_csv(), load_dvf_plus() (enriched DVF+)
  analyze.py      # summarize_mutations(), price_stats()
  query.py        # QueryParser, QueryExecutor for structured queries

app.py            # Streamlit chat interface for querying DVF data

scripts/          # Data acquisition + Marseille dataset generation
  download_dvf_sample.py                  # Fetch raw DVF data
  build_marseille_houses_sold_on_date.py  # → data/processed/marseille_houses_sold_<date>.csv

config/defaults.yaml  # Paths, encoding (utf-8), separator ("|")
```

**Data pipeline**: `data/raw/ValeursFoncieres-2025.txt` → Marseille houses sold on the target date (default 15/09/2025) → enriched with each parcel's prior-sale history (JSON `previous_mutations`, scanned across all `data/raw/ValeursFoncieres-*.txt`) → `data/processed/marseille_houses_sold_<YYYY-MM-DD>.csv` (consumed by vision360immeuble). See README "Marseille houses-sold dataset".

## Key Conventions

- **Package manager**: UV (not pip). Always use `uv run` or `uv sync`.
- **Python**: 3.11+
- **Raw DVF files**: pipe-separated (`|`), not comma-separated.
- **Line length**: 100 (ruff config in pyproject.toml)
- **Build backend**: hatchling
- **Source layout**: `src/dvf/` (configured as pythonpath in pytest)
- **Comments**: Use comments sparingly. Only add comments where the logic isn't self-evident.

# ignore this comment