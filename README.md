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

3. **Notebooks**: From the **project root**, run `uv run jupyter lab` and open `notebooks/`. The notebook adds the project `src` to the path so `dvf` is found whether the kernel cwd is the project root or the `notebooks/` folder.

## Chat interface (Streamlit)

The Streamlit app (`app.py`) offers two sidebar modes:

| Mode | Data scope | Typical use |
|------|----------------|-------------|
| **Structured query** | France-wide grouped houses (`df_grouped_2020_2025_france_cleaned.csv`) | Count / mean / median / min / max prices with postal code, commune, surface filters |
| **RAG (natural language)** | **Marseille houses only** — all arrondissements, INSEE **13201–13216** (`marseille_houses_dvf.csv` + Chroma index) | Semantic Q&A (“cheap houses”, “large villas”, themes not mapped to strict filters) |

RAG answers only reflect whatever was indexed into `data/vectorstore_marseille/`. Structured queries run over the full France file (including Marseille rows).

### Setup

1. **Dependencies** (includes Streamlit, OpenAI, ChromaDB):

   ```bash
   uv sync
   ```

2. **Structured mode — France dataset**

   - Path: `data/processed/df_grouped_2020_2025_france_cleaned.csv`
   - Produced by the cleaning pipeline (e.g. `notebooks/03_dvf_clean_duplicates_houses.ipynb`).
   - Processed CSVs often store **`Code postal` as numbers** (e.g. `13008.0`). The query layer compares numerically so filters like `13008` still match.

3. **RAG mode — Marseille-only subset**

   - Tabular slice: `data/processed/marseille_houses_dvf.csv`  
     (houses in Marseille / all arrondissements, INSEE 13201–13216; build via `scripts/extract_marseille_houses.py`).
   - Vector index (required for RAG): build into `data/vectorstore_marseille/`:

     ```bash
     export OPENAI_API_KEY=...   # required for embeddings
     uv run python scripts/build_vectorstore.py -n 9144   # full Marseille slice (~9,144 rows); omit -n only after confirming cost/time
     ```

   The app enables RAG only if that CSV exists **and** `data/vectorstore_marseille/` contains a persisted Chroma DB (not just `.gitkeep`).

   If RAG answers omit prices after a code update, **rebuild** the vector store so each chunk includes parsed mutation amounts (clear `data/vectorstore_marseille/*` except dotfiles like `.gitkeep` if you use it, then run the command above). Details: `RAG_README.md`.

4. **API key**

   - Set `OPENAI_API_KEY` in the environment or project-root `.env` (see `.env.example`).
   - Used for: optional structured-query parsing (LLM), RAG embeddings + generation. Without a key, structured mode falls back to regex parsing; RAG stays disabled until embeddings have been built (they still required a key at build time).

### Run

```bash
uv run streamlit run app.py
```

Open `http://localhost:8501`.

### Example prompts

**Structured (France)**

- “What is the mean price of a 100m² house in 13008 Marseille?”
- “How many houses are in Marseille?”
- “What is the median price of houses in Paris?”

**RAG (Marseille houses only)**

- “What are the most expensive houses in Marseille?”
- “Quelles sont les maisons les moins chères à Marseille?”

Use Marseille-related wording in RAG; Paris-only questions are outside the indexed corpus.

### Behaviour summary

Structured mode extracts **postal code**, **commune** (substring match on official commune labels), **surface** (±10%), and **query type** (mean, median, count, min, max). Results include counts, aggregates, average €/m² and average surface where prices exist.

More detail on RAG indexing and troubleshooting: `RAG_README.md`.

## Commands

| Command | Description |
|--------|-------------|
| `uv sync` | Create/update venv and install deps from `pyproject.toml` |
| `uv add <pkg>` | Add a dependency |
| `uv run python script.py` | Run script in project env |
| `uv run pytest` | Run tests |
| `uv run jupyter lab` | Start Jupyter Lab |
| `uv run streamlit run app.py` | Launch the chat interface |

## Licence

Code: your choice. DVF data: [Licence Ouverte / Open Licence v2.0](https://www.etalab.gouv.fr/licence-ouverte-open-licence/) — respect reuse and non-reidentification rules.

## Resources
Conceptual Data Model: https://www.groupe-dvf.fr/vademecum-fiche-n3-precautions-techniques-et-qualite-des-donnees-dvf/
