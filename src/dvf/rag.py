"""RAG (Retrieval-Augmented Generation) system for DVF dataset queries."""

import ast
import json
import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv

try:
    import chromadb
    from chromadb.config import Settings
    from openai import OpenAI
except ImportError:
    chromadb = None
    OpenAI = None

# Load .env from project root
project_root = Path(__file__).parent.parent.parent
env_path = project_root / ".env"
load_dotenv(dotenv_path=env_path)

logger = logging.getLogger(__name__)


class DVFVectorStore:
    """Vector store for DVF dataset using ChromaDB."""

    def __init__(self, collection_name: str = "dvf_properties", persist_directory: str | None = None):
        """Initialize vector store.

        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the database (default: data/vectorstore)
        """
        if chromadb is None:
            raise ImportError("chromadb is required. Install with: uv add chromadb")

        if persist_directory is None:
            persist_directory = project_root / "data" / "vectorstore"
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False),
        )

        try:
            self.collection = self.client.get_collection(name=collection_name)
            logger.info(f"Loaded existing collection: {collection_name}")
        except Exception:
            self.collection = None
            logger.info(f"Collection {collection_name} does not exist yet")

        self.openai_client = None
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key and OpenAI is not None:
            self.openai_client = OpenAI(api_key=api_key)

    def _create_text_representation(self, row: pd.Series) -> str:
        """Create a text representation of a property row for embedding."""
        parts = []

        if pd.notna(row.get("Commune")):
            parts.append(f"Commune: {row['Commune']}")
        if pd.notna(row.get("Code postal")):
            parts.append(f"Code postal: {row['Code postal']}")
        if pd.notna(row.get("Type local")):
            parts.append(f"Type: {row['Type local']}")
        if pd.notna(row.get("Surface reelle bati")):
            parts.append(f"Surface: {row['Surface reelle bati']:.0f} m²")
        if pd.notna(row.get("Nombre pieces principales")):
            parts.append(f"Pieces: {row['Nombre pieces principales']:.0f}")
        if pd.notna(row.get("Voie")):
            parts.append(f"Adresse: {row['Voie']}")
        if pd.notna(row.get("mutations")):
            try:
                import ast
                mutations_str = row["mutations"]
                if isinstance(mutations_str, str):
                    mutations = ast.literal_eval(mutations_str)
                else:
                    mutations = mutations_str
                    
                if mutations and len(mutations) > 0:
                    if isinstance(mutations, list) and len(mutations) > 0:
                        last_mutation = mutations[-1]
                    elif isinstance(mutations, dict):
                        last_mutation = list(mutations.values())[-1]
                    else:
                        last_mutation = None
                        
                    if isinstance(last_mutation, dict):
                        price = list(last_mutation.values())[0]
                        if isinstance(price, (int, float)):
                            parts.append(f"Dernier prix: {price:,.0f} €")
            except Exception:
                pass

        return ". ".join(parts)

    def _get_embedding(self, text: str) -> list[float]:
        """Get embedding for text using OpenAI."""
        if not self.openai_client:
            raise ValueError("OpenAI client not initialized. Set OPENAI_API_KEY.")

        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=text,
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Failed to get embedding: {e}")
            raise

    def _get_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings for multiple texts in a single API call (more efficient)."""
        if not self.openai_client:
            raise ValueError("OpenAI client not initialized. Set OPENAI_API_KEY.")

        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=texts,
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error(f"Failed to get embeddings batch: {e}")
            raise

    def index_dataframe(
        self, 
        df: pd.DataFrame, 
        batch_size: int = 1000, 
        sample_size: int | None = None,
        progress_callback: Callable[[int, int], None] | None = None
    ) -> None:
        """Index DataFrame into vector store.

        Args:
            df: DataFrame to index
            batch_size: Number of rows to process in each batch (also used for embedding API calls)
            sample_size: If provided, only index a sample of this size (for testing)
            progress_callback: Optional callback function(current, total) for progress updates
        """
        if self.openai_client is None:
            raise ValueError("OpenAI client not initialized. Set OPENAI_API_KEY.")

        if sample_size:
            df = df.sample(n=min(sample_size, len(df)), random_state=42)
            logger.info(f"Indexing sample of {len(df)} rows")

        if self.collection is None:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "DVF French real estate properties"},
            )

        total_rows = len(df)
        logger.info(f"Indexing {total_rows} properties...")

        texts = []
        metadatas = []
        ids = []
        processed = 0

        for idx, (_, row) in enumerate(df.iterrows()):
            text = self._create_text_representation(row)
            texts.append(text)

            metadata = {
                "row_index": str(idx),
                "commune": str(row.get("Commune", "")),
                "postal_code": str(row.get("Code postal", "")),
                "type_local": str(row.get("Type local", "")),
            }
            metadatas.append(metadata)
            ids.append(f"property_{idx}")

            # Process batch when we reach batch_size
            if len(texts) >= batch_size:
                # Get embeddings in batch (more efficient than one-by-one)
                embeddings = self._get_embeddings_batch(texts)
                self.collection.add(
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=metadatas,
                    ids=ids,
                )
                processed += len(texts)
                
                if progress_callback:
                    progress_callback(processed, total_rows)
                
                texts = []
                metadatas = []
                ids = []

        # Process remaining items
        if texts:
            embeddings = self._get_embeddings_batch(texts)
            self.collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids,
            )
            processed += len(texts)
            if progress_callback:
                progress_callback(processed, total_rows)

        logger.info(f"Indexed {total_rows} properties successfully")

    def search(self, query: str, n_results: int = 10, filter_dict: dict[str, Any] | None = None) -> list[dict]:
        """Search for similar properties.

        Args:
            query: Natural language query
            n_results: Number of results to return
            filter_dict: Optional filters (e.g., {"commune": "Paris"})

        Returns:
            List of search results with metadata
        """
        if self.collection is None:
            raise ValueError("Collection not initialized. Call index_dataframe first.")

        if self.openai_client is None:
            raise ValueError("OpenAI client not initialized. Set OPENAI_API_KEY.")

        query_embedding = self._get_embedding(query)

        where_clause = None
        if filter_dict:
            where_clause = filter_dict

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_clause,
        )

        formatted_results = []
        for i in range(len(results["ids"][0])):
            formatted_results.append({
                "id": results["ids"][0][i],
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i] if "distances" in results else None,
            })

        return formatted_results


class DVFRAGSystem:
    """RAG system for querying DVF dataset with natural language."""

    def __init__(self, vector_store: DVFVectorStore, df: pd.DataFrame, model: str = "gpt-4o-mini"):
        """Initialize RAG system.

        Args:
            vector_store: Initialized vector store
            df: Original DataFrame for retrieving full row data
            model: OpenAI model to use for generation
        """
        self.vector_store = vector_store
        self.df = df
        self.model = model

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")
        self.openai_client = OpenAI(api_key=api_key)

    def query(self, user_query: str, language: str = "auto", max_results: int = 10) -> dict[str, Any]:
        """Query the dataset using natural language.

        Args:
            user_query: Natural language query in English or French
            language: Response language ("en", "fr", or "auto" to detect)
            max_results: Maximum number of retrieved documents

        Returns:
            Dictionary with answer, sources, and metadata
        """
        logger.info(f"RAG query: {user_query}")

        # Detect language if auto
        if language == "auto":
            language = "fr" if any(char in user_query for char in "àéèêëîïôùûüç") else "en"

        # Search for relevant properties
        search_results = self.vector_store.search(user_query, n_results=max_results)

        if not search_results:
            return {
                "answer": "Aucune propriété trouvée correspondant à votre requête." if language == "fr" else "No properties found matching your query.",
                "sources": [],
                "metadata": {"results_count": 0},
            }

        # Build context from search results
        context_parts = []
        for i, result in enumerate(search_results[:max_results], 1):
            context_parts.append(f"[Property {i}]\n{result['document']}")

        context = "\n\n".join(context_parts)

        # Generate answer using LLM
        system_prompt_en = """You are a helpful assistant answering questions about French real estate data (DVF - Demandes de Valeurs Foncières).

Use the provided context about properties to answer the user's question accurately. If the context doesn't contain enough information, say so.

Format your response clearly with:
- A direct answer to the question
- Relevant statistics or numbers if applicable
- Any important caveats or limitations

Be concise but informative."""

        system_prompt_fr = """Vous êtes un assistant utile qui répond aux questions sur les données immobilières françaises (DVF - Demandes de Valeurs Foncières).

Utilisez le contexte fourni sur les propriétés pour répondre précisément à la question de l'utilisateur. Si le contexte ne contient pas assez d'informations, dites-le.

Formatez votre réponse clairement avec :
- Une réponse directe à la question
- Des statistiques ou nombres pertinents si applicable
- Toute mise en garde ou limitation importante

Soyez concis mais informatif."""

        system_prompt = system_prompt_fr if language == "fr" else system_prompt_en

        user_prompt = f"""Context about properties:
{context}

User question: {user_query}

Answer the question based on the context provided."""

        try:
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
            )

            answer = response.choices[0].message.content

            return {
                "answer": answer,
                "sources": search_results,
                "metadata": {
                    "results_count": len(search_results),
                    "language": language,
                    "model": self.model,
                },
            }
        except Exception as e:
            logger.error(f"Failed to generate answer: {e}", exc_info=True)
            return {
                "answer": f"Erreur lors de la génération de la réponse: {e}" if language == "fr" else f"Error generating answer: {e}",
                "sources": search_results,
                "metadata": {"error": str(e)},
            }
