# DVF Data Analysis

Data pipeline for building clean **Marseille house datasets** from **DVF** (Demandes de Valeurs Foncières) data — French open data on real estate mutations (sales, etc.) from notarial deeds. The processed CSVs it produces are consumed by a separate downstream project, **vision360immeuble**. A Streamlit app is included for ad-hoc structured querying of the data.

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
- **Run without activating**: `uv run python script.py`.

## Project layout

```
ldi_dvf/
├── data/
│   ├── raw/          # DVF / DVF+ files (.txt, .csv) — place downloads here
│   └── processed/    # Cleaned / aggregated outputs (Marseille datasets)
├── scripts/          # Data download + Marseille dataset generation
├── src/
│   └── dvf/          # Package: load, analyze, query
├── app.py            # Streamlit chat interface
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

3. **Generate the Marseille dataset**: From the **project root**, run the scripts in order:
   ```bash
   uv run python scripts/extract_marseille_houses.py        # France → Marseille houses slice
   uv run python scripts/group_marseille_house_mutations.py # → data/processed/marseille_houses_grouped.csv
   ```
   The grouped CSV is the entry point of the [Marseille repeat-sales pipeline](#marseille-repeat-sales-pipeline) and the dataset consumed by **vision360immeuble**.

## Chat interface (Streamlit)

The Streamlit app (`app.py`) runs **structured queries** over the France-wide grouped houses
(`df_grouped_2020_2025_france_cleaned.csv`): count / mean / median / min / max prices with
postal-code, commune, and surface filters (Marseille rows included). Query parsing uses an LLM
when `OPENAI_API_KEY` is set, and falls back to regex otherwise.

### Setup

1. **Dependencies** (includes Streamlit and OpenAI):

   ```bash
   uv sync
   ```

2. **Dataset**

   - Path: `data/processed/df_grouped_2020_2025_france_cleaned.csv`
   - Produced by the cleaning/grouping pipeline.
   - Processed CSVs often store **`Code postal` as numbers** (e.g. `13008.0`). The query layer compares numerically so filters like `13008` still match.

3. **API key**

   - Set `OPENAI_API_KEY` in the environment or project-root `.env` (see `.env.example`).
   - Used for optional LLM-powered query parsing. Without a key, parsing falls back to regex.

### Run

```bash
uv run streamlit run app.py
```

Open `http://localhost:8501`.

### Example prompts

- “What is the mean price of a 100m² house in 13008 Marseille?”
- “How many houses are in Marseille?”
- “What is the median price of houses in Paris?”

### Behaviour summary

The app extracts **postal code**, **commune** (substring match on official commune labels), **surface** (±10%), and **query type** (mean, median, count, min, max). Results include counts, aggregates, average €/m² and average surface where prices exist.

## Marseille repeat-sales pipeline

A staged filtering pipeline that narrows the raw Marseille house data down to a clean
repeat-sales dataset suitable for modelling property-price appreciation. Each stage is a
CSV in `data/processed/`, progressively stricter:

| # | File | Rows | What it contains |
|---|------|------|------------------|
| 1 | `marseille_houses_grouped.csv` | 7,464 parcels | **Source.** One row per cadastral parcel/address; each row holds a JSON list of that parcel's mutations (8,426 total, 2020–2025, all 16 arrondissements). |
| 2 | `marseille_houses_multi_mutations.csv` | 542 | Parcels with **>1 mutation**. |
| 3 | `marseille_houses_repeat_sales.csv` | 494 units | Same parcel **+ same `surface_reelle_bati`**, sold ≥2×. Splits multi-unit residences by surface, but identical-floorplan units still collide. |
| 4 | `marseille_houses_true_repeat_sales.csv` | 288 dwellings | **Strict single dwelling:** exactly one surface across all mutations, `Vente` only, 2–3 sales, ≥180 days between consecutive sales, positive prices. Isolates one physical house resold over time. |
| 5 | `marseille_houses_repeat_sales_model_ready.csv` | 235 dwellings | **Model-ready.** Stage 4 + outlier trim: annualized appreciation within a robust IQR band (≈[−30%, +59%]/yr) and €/m² within the 1st–99th percentile on both sales. Removes data-entry errors and land-assembly/renovation jumps. |

**Why the stages matter:** the high mutation counts at stages 2–3 are residences /
co-ownerships (many units share one parcel), *not* one house resold dozens of times —
varying (or coincidentally identical) surfaces give them away. Stages 4–5 enforce a
genuine single-dwelling repeat-sale so price changes reflect real appreciation. After the
trim, median appreciation is **~10%/yr** with realistic €/m² — economically sensible for
Marseille 2020–2025.

**Model-ready columns:** parcel/address identifiers, `surface_reelle_bati`, `n_sales`,
`first_date`/`last_date`, `min_gap_days`, `years`, `first_price`/`last_price`,
`ppm2_first`/`ppm2_last`, `pct_change` (total), `ann_pct` (annualized), and the raw
`mutations` JSON.

**Caveat:** 235 pairs is small for RandomForest. Use this file when the target is
per-dwelling **price appreciation** (`ann_pct`); for absolute-price prediction prefer the
larger cross-sectional France dataset.

## Commands

| Command | Description |
|--------|-------------|
| `uv sync` | Create/update venv and install deps from `pyproject.toml` |
| `uv add <pkg>` | Add a dependency |
| `uv run python script.py` | Run script in project env |
| `uv run pytest` | Run tests |
| `uv run python scripts/extract_marseille_houses.py` | Build the Marseille houses slice |
| `uv run python scripts/group_marseille_house_mutations.py` | Group mutations → `marseille_houses_grouped.csv` |
| `uv run streamlit run app.py` | Launch the chat interface |

## Licence

Code: your choice. DVF data: [Licence Ouverte / Open Licence v2.0](https://www.etalab.gouv.fr/licence-ouverte-open-licence/) — respect reuse and non-reidentification rules.

## Resources
Conceptual Data Model: https://www.groupe-dvf.fr/vademecum-fiche-n3-precautions-techniques-et-qualite-des-donnees-dvf/
