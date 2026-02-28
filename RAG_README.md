# RAG System for DVF Dataset

This project includes a RAG (Retrieval-Augmented Generation) system that allows you to query the DVF dataset using natural language in English or French.

## Overview

The RAG system combines:
- **Vector Database** (ChromaDB): Stores embeddings of property descriptions
- **Semantic Search**: Finds relevant properties based on query similarity
- **LLM Generation** (OpenAI GPT): Generates natural language answers based on retrieved context

## Setup

### 1. Install Dependencies

```bash
uv sync
```

This will install `chromadb` and `tiktoken` required for the RAG system.

### 2. Build Vector Store

Before using RAG mode, you need to build the vector store from your dataset:

```bash
uv run python scripts/build_vectorstore.py
```

**Note**: This process:
- Uses OpenAI embeddings API (requires `OPENAI_API_KEY`)
- Can take a while for large datasets (2.5M+ rows)
- For testing, you can modify the script to use `sample_size=10000` first

The vector store is saved to `data/vectorstore_ensues/` (Ensues houses only).

### 3. Configure API Key

Make sure `OPENAI_API_KEY` is set in your `.env` file:

```bash
OPENAI_API_KEY=your_key_here
```

## Usage

### Streamlit App

1. Launch the app:
   ```bash
   uv run streamlit run app.py
   ```

2. In the sidebar, select **"RAG (Natural Language)"** query mode

3. Ask questions in English or French:
   - "Quelles sont les maisons les moins chères à Paris?"
   - "Find me houses with garden in Marseille"
   - "What are the most expensive properties in Ensues?"
   - "Combien coûte en moyenne une maison de 100m²?"

### Python API

```python
from src.dvf.rag import DVFVectorStore, DVFRAGSystem
import pandas as pd

# Load your dataset
df = pd.read_csv("data/processed/df_grouped_2020_2025_france_cleaned.csv")

# Initialize vector store (loads existing if available)
vector_store = DVFVectorStore()

# Initialize RAG system
rag_system = DVFRAGSystem(vector_store, df)

# Query
result = rag_system.query("What are the cheapest houses in Paris?", language="en")
print(result["answer"])
```

## Architecture

### Components

1. **DVFVectorStore**: Manages ChromaDB collection
   - Creates embeddings using OpenAI `text-embedding-3-small`
   - Stores property descriptions as text documents
   - Enables semantic search

2. **DVFRAGSystem**: Orchestrates RAG pipeline
   - Searches vector store for relevant properties
   - Builds context from retrieved results
   - Generates answers using GPT models

### Data Representation

Each property is converted to a text description including:
- Commune name
- Postal code
- Property type
- Surface area
- Number of rooms
- Address
- Latest transaction price

## Query Modes Comparison

| Feature | Structured Query | RAG Mode |
|---------|------------------|----------|
| Query Type | Parameter extraction | Natural language |
| Languages | English | English + French |
| Response | Aggregated statistics | Natural language answer |
| Use Case | Specific filters | Exploratory questions |
| Speed | Fast | Slower (API calls) |

## Performance Notes

- **Indexing**: ~500 properties/second (depends on API rate limits)
- **Query**: ~2-5 seconds per query (embedding + LLM generation)
- **Storage**: Vector store size ~100-200MB per million properties

## Troubleshooting

### Vector store not found
- Run `scripts/build_vectorstore.py` first (builds from `df_2020_2025_houses_ensues.csv`)
- Check that `data/vectorstore_ensues/` directory exists

### API errors
- Verify `OPENAI_API_KEY` is set correctly
- Check API rate limits and quotas
- For large datasets, consider indexing in batches

### Slow queries
- Reduce `max_results` parameter
- Use a smaller sample for testing
- Consider using a faster embedding model

# minor change main