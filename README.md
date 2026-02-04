# DVF Data Analysis

Data science project for analyzing **DVF** (Demandes de Valeurs Foncières) data — French open data on real estate mutations (sales, etc.) from notarial deeds.

- **Official DVF**: [data.gouv.fr – Demandes de valeurs foncières](https://www.data.gouv.fr/fr/datasets/demandes-de-valeurs-foncieres/)
- **DVF+** (enriched, geolocated): [data.gouv.fr – DVF+ open-data](https://www.data.gouv.fr/fr/datasets/dvf-open-data/)

Data is updated semestrially (April / October). Format: pipe-separated `.txt` (official) or CSV (DVF+). **Personal data**: do not re-identify or index for external search; respect the open licence terms.

## Setup with UV

[UV](https://docs.astral.sh/uv/) is used for virtual env and dependency management.

```bash
# Create venv and install dependencies (from project root)
uv sync

# Or with optional dev + geo deps
uv sync --extra dev --extra geo
```

- **First run**: UV creates `.venv` and `uv.lock`.
- **Activate** (optional): `source .venv/bin/activate` (Unix) or `.venv\Scripts\activate` (Windows).
- **Run without activating**: `uv run python script.py` or `uv run jupyter lab`.

## Project layout

```
ldi_dvf/
├── data/
│   ├── raw/          # DVF / DVF+ files (.txt, .csv) — place downloads here
│   └── processed/    # Cleaned / aggregated outputs
├── notebooks/        # Jupyter notebooks for exploration
├── scripts/          # One-off scripts (download, pipelines)
├── src/
│   └── dvf/          # Package: load, analyze
├── tests/
├── config/           # Optional config (e.g. defaults.yaml)
├── pyproject.toml
├── .python-version   # 3.11
└── README.md
```

## Quick start

1. **Get data**: Download a DVF or DVF+ file from data.gouv.fr and put it in `data/raw/`.

2. **Load in Python**:
   ```python
   from dvf import load_dvf_raw, summarize_mutations

   df = load_dvf_raw("data/raw/dvf_2023.txt", nrows=10_000)  # sample for testing
   summary = summarize_mutations(df)
   ```

3. **Notebooks**: From the **project root**, run `uv run jupyter lab` and open `notebooks/`. The notebook adds the project `src` to the path so `dvf` is found whether the kernel cwd is the project root or the `notebooks/` folder.

## Commands

| Command | Description |
|--------|-------------|
| `uv sync` | Create/update venv and install deps from `pyproject.toml` |
| `uv add <pkg>` | Add a dependency |
| `uv run python script.py` | Run script in project env |
| `uv run pytest` | Run tests |
| `uv run jupyter lab` | Start Jupyter Lab |

## Licence

Code: your choice. DVF data: [Licence Ouverte / Open Licence v2.0](https://www.etalab.gouv.fr/licence-ouverte-open-licence/) — respect reuse and non-reidentification rules.
