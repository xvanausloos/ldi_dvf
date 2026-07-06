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

3. **Generate the Marseille dataset**: From the **project root**, run:
   ```bash
   uv run python scripts/build_marseille_houses_sold_on_date.py               # default 15/09/2025
   uv run python scripts/build_marseille_houses_sold_on_date.py --date 15/09/2025
   ```
   This produces `data/processed/marseille_houses_sold_<YYYY-MM-DD>.csv`, the dataset consumed
   by **vision360immeuble**. See [Marseille houses-sold dataset](#marseille-houses-sold-dataset).

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

## Marseille houses-sold dataset

`scripts/build_marseille_houses_sold_on_date.py` builds one dataset: **every house (Maison)
sold in Marseille on a target date** (default **15/09/2025**), enriched with the prior sale
history of each house's cadastral parcel.

- **Primary rows** come from a single file, `data/raw/ValeursFoncieres-2025.txt`: Marseille
  arrondissements (INSEE 13201–13216) × `Code type local == 1` (Maison) × `Date mutation == target`.
- **History** is added by scanning **all** `data/raw/ValeursFoncieres-*.txt` (2020–2025) for
  earlier mutations of the same parcel, stored as a JSON string in `previous_mutations`.
- **Date** is configurable: `--date dd/mm/YYYY` (default `15/09/2025`).
- **Output**: `data/processed/marseille_houses_sold_<YYYY-MM-DD>.csv` — consumed by **vision360immeuble**.

"Same house" = unique cadastral parcel `insee_code + Prefixe de section + Section + No plan`.
Multiple lot rows of one sale are collapsed into a single mutation.

**Columns**

| Column | Example | Notes |
|--------|---------|-------|
| `insee_code`, `Commune`, `code_postal` | `13207`, `MARSEILLE 7EME`, `13007` | arrondissement identity |
| `adresse` | `130 RUE DU VALLON DES AUFFES, 13007 MARSEILLE 7EME` | ready-to-paste full address |
| `prefixe_section`, `Section`, `No plan` | `830`, `A`, `13` | cadastral parcel parts |
| `id_parcelle` | `132078300A0013` | 14-char id = `code_commune(5)+prefixe(3)+section(2)+no_plan(4)`, key used by DVF géolocalisé |
| `No voie`, `btq`, `Type de voie`, `Voie` | `130`, ``, `RUE`, `DU VALLON DES AUFFES` | address parts |
| `date_mutation`, `nature_mutation`, `valeur_fonciere` | `2025-09-15`, `Vente`, `725000.0` | the target-date sale |
| `surface_reelle_bati`, `nombre_pieces_principales`, `surface_terrain` | `83.0`, `2.0`, `` | physical detail of the sale |
| `n_previous_mutations` | `1` | count of earlier sales of the parcel |
| `previous_mutations` | `[{"date_mutation": "2025-05-12", "valeur_fonciere": 525000.0}]` | JSON history (date + price), chronological |

Verify any sale on the official [explorer](https://explore.data.gouv.fr/fr/immobilier) using
`code_postal` / `adresse` / `id_parcelle`.

## Commands

| Command | Description |
|--------|-------------|
| `uv sync` | Create/update venv and install deps from `pyproject.toml` |
| `uv add <pkg>` | Add a dependency |
| `uv run python script.py` | Run script in project env |
| `uv run pytest` | Run tests |
| `uv run python scripts/build_marseille_houses_sold_on_date.py` | Build the houses-sold dataset (default 15/09/2025) |
| `uv run python scripts/build_marseille_houses_sold_on_date.py --date 15/09/2025` | Same, for an explicit date |
| `uv run streamlit run app.py` | Launch the chat interface |

## Licence

Code: your choice. DVF data: [Licence Ouverte / Open Licence v2.0](https://www.etalab.gouv.fr/licence-ouverte-open-licence/) — respect reuse and non-reidentification rules.

## Resources
Conceptual Data Model: https://www.groupe-dvf.fr/vademecum-fiche-n3-precautions-techniques-et-qualite-des-donnees-dvf/
