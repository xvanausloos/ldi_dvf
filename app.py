"""Streamlit chat interface for querying DVF real estate data."""

import logging
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent / "src"))

from dvf.query import QueryExecutor, QueryParser

try:
    from dvf.rag import DVFVectorStore, DVFRAGSystem
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

# Load .env from project root
project_root = Path(__file__).parent
env_path = project_root / ".env"
load_dotenv(dotenv_path=env_path)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
if not RAG_AVAILABLE:
    logger.warning("RAG system not available. Install chromadb: uv add chromadb")

# Verify API key is loaded
if os.getenv("OPENAI_API_KEY"):
    logger.info("OPENAI_API_KEY loaded successfully")
else:
    logger.warning("OPENAI_API_KEY not found in environment")

DATA_PATH = Path(__file__).parent / "data" / "processed" / "df_grouped_2020_2025_france_cleaned.csv"
RAG_DATA_PATH = Path(__file__).parent / "data" / "processed" / "df_2020_2025_houses_ensues.csv"


@st.cache_data
def load_data():
    """Load and prepare DVF data (full France) for structured queries."""
    if not DATA_PATH.exists():
        st.error(f"Data file not found: {DATA_PATH}")
        st.stop()
    df = pd.read_csv(DATA_PATH, low_memory=False)
    return df


@st.cache_data
def load_rag_data():
    """Load Ensues-only dataset for RAG (natural language queries)."""
    if not RAG_DATA_PATH.exists():
        return None
    return pd.read_csv(RAG_DATA_PATH, low_memory=False)


def format_result(result: dict) -> str:
    """Format query result as a readable string."""
    if not result.get("success"):
        return f"❌ {result.get('message', 'Query failed')}"

    parts = [f"✅ **{result['label']}**: {result['value']:,.0f} {result['unit']}"]
    parts.append(f"\n📊 Found {result['count']:,} properties matching your criteria")

    if "price_per_m2" in result and pd.notna(result["price_per_m2"]):
        parts.append(f"\n💰 Average price per m²: {result['price_per_m2']:,.0f} €/m²")

    if "surface_avg" in result and pd.notna(result["surface_avg"]):
        parts.append(f"\n📐 Average surface: {result['surface_avg']:.1f} m²")

    return "\n".join(parts)


def process_query(
    prompt: str, executor: QueryExecutor, parser: QueryParser
) -> tuple[str, dict, dict, str, bool]:
    """Process a query and return the formatted response, result dict, params dict, SQL query, and LLM usage flag."""
    params, used_llm = parser.parse(prompt)
    result = executor.execute(params)
    sql_query = executor.generate_sql(params)
    return format_result(result), result, params, sql_query, used_llm


def main():
    """Main Streamlit app."""
    st.set_page_config(page_title="DVF Data Chat", page_icon="🏠", layout="wide")

    st.title("🏠 DVF Data Chat")
    st.markdown("Ask questions about French real estate data (2020-2025)")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "pending_query" not in st.session_state:
        st.session_state.pending_query = None

    if "df" not in st.session_state:
        with st.spinner("Loading data..."):
            st.session_state.df = load_data()
            st.session_state.executor = QueryExecutor(st.session_state.df)
            st.session_state.parser = QueryParser(use_llm=True, df=st.session_state.df)
            
            # Initialize RAG system if available (uses Ensues-only dataset)
            if RAG_AVAILABLE:
                df_ensues = load_rag_data()
                vectorstore_path = project_root / "data" / "vectorstore_ensues"
                if df_ensues is not None and vectorstore_path.exists() and any(vectorstore_path.iterdir()):
                    try:
                        st.session_state.vector_store = DVFVectorStore(
                            persist_directory=str(vectorstore_path)
                        )
                        st.session_state.rag_system = DVFRAGSystem(
                            st.session_state.vector_store,
                            df_ensues,
                        )
                        st.session_state.rag_available = True
                        st.session_state.rag_df = df_ensues
                    except Exception as e:
                        logger.warning(f"Failed to initialize RAG system: {e}")
                        st.session_state.rag_available = False
            else:
                st.session_state.rag_available = False
            else:
                st.session_state.rag_available = False

    with st.sidebar:
        st.header("⚙️ Controls")
        if st.button("🗑️ Clear Chat History", use_container_width=True, type="secondary"):
            st.session_state.messages = []
            st.session_state.pending_query = None
            st.rerun()

        st.divider()

        st.header("🔍 Query Mode")
        query_mode = st.radio(
            "Select query mode",
            ["Structured Query", "RAG (Natural Language)"],
            index=0,
            help="Structured Query: Extract parameters and filter data. RAG: Natural language Q&A with semantic search.",
        )
        
        use_rag = query_mode == "RAG (Natural Language)"
        
        if use_rag and not st.session_state.get("rag_available", False):
            st.error(
                "⚠️ RAG mode requires the Ensues vector store. "
                "Run: `uv run python scripts/build_vectorstore.py` (builds from "
                "df_2020_2025_houses_ensues.csv → data/vectorstore_ensues/)"
            )
            use_rag = False
        if use_rag:
            st.caption("📌 RAG interroge uniquement les maisons d’Ensues (13820).")

        st.divider()

        st.header("🤖 LLM Settings")
        use_llm = st.toggle(
            "Use LLM for query parsing",
            value=True,
            help="Enable LLM-powered query understanding (requires OPENAI_API_KEY). Falls back to regex if disabled or unavailable.",
        )
        if use_llm:
            model = st.selectbox(
                "Model",
                ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
                index=0,
                help="OpenAI model to use for query parsing",
            )
        else:
            model = "gpt-4o-mini"

        if use_llm:
            import os

            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                st.warning(
                    "⚠️ OPENAI_API_KEY not found. LLM parsing will fall back to regex. "
                    "Set OPENAI_API_KEY in your environment or .env file."
                )
            else:
                st.success(f"✅ LLM enabled (model: {model})")
                
            # Debug info
            if st.checkbox("Show debug info", value=False):
                st.text(f"use_llm flag: {use_llm}")
                st.text(f"API key present: {bool(api_key)}")
                if api_key:
                    st.text(f"API key length: {len(api_key)}")
                if "parser" in st.session_state:
                    parser = st.session_state.parser
                    st.text(f"Parser.use_llm: {parser.use_llm}")
                    st.text(f"Parser.client exists: {parser.client is not None}")
                    st.text(f"Parser.model: {parser.model}")
                    if parser.semantic_layer:
                        st.text(f"Semantic layer: {len(parser.semantic_layer)} chars")
                else:
                    st.text("Parser not initialized")

        st.divider()

        st.header("💡 Example Queries")
        if use_rag:
            examples = [
                "Quelles sont les maisons les moins chères à Paris?",
                "Trouve-moi des maisons avec jardin à Marseille",
                "What are the most expensive houses in Ensues?",
                "Combien coûte en moyenne une maison de 100m²?",
            ]
        else:
            examples = [
                "What is the mean price of a 100m² house in 13820 Ensues?",
                "How many houses are in Marseille?",
                "What is the median price of houses in Paris?",
                "What is the average price of a 80m² house in 75001?",
            ]
        for example in examples:
            if st.button(example, key=f"example_{hash(example)}", use_container_width=True):
                st.session_state.pending_query = example
                st.rerun()

        st.divider()

        st.header("📊 Dataset Info")
        if use_rag and st.session_state.get("rag_df") is not None:
            st.caption("RAG: maisons Ensues uniquement")
            st.metric("Properties (Ensues)", f"{len(st.session_state.rag_df):,}")
        else:
            st.metric("Total Properties", f"{len(st.session_state.df):,}")
            st.metric(
                "With Price Data",
                f"{st.session_state.df['mutations'].notna().sum():,}",
            )

    if "parser" in st.session_state:
        current_use_llm = getattr(st.session_state.parser, "use_llm", False)
        current_model = getattr(st.session_state.parser, "model", "gpt-4o-mini")
        if current_use_llm != use_llm or (use_llm and current_model != model):
            st.session_state.parser = QueryParser(use_llm=use_llm, model=model, df=st.session_state.df)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                if "query_details" in message:
                    with st.expander("📋 Query details"):
                        st.json(message["query_details"])
                if "sql_query" in message:
                    with st.expander("🗄️ SQL Query"):
                        st.code(message["sql_query"], language="sql")

    prompt = None
    if st.session_state.pending_query:
        prompt = st.session_state.pending_query
        st.session_state.pending_query = None
    elif user_input := st.chat_input("Ask a question about real estate data..."):
        prompt = user_input

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            if use_rag and st.session_state.get("rag_available", False):
                # RAG mode
                with st.spinner("🔍 Searching properties and generating answer..."):
                    rag_result = st.session_state.rag_system.query(
                        prompt, 
                        language="auto",
                        max_results=10
                    )
                    
                    st.success("🤖 Answered using RAG (Retrieval-Augmented Generation)")
                    st.markdown(rag_result["answer"])
                    
                    if rag_result.get("sources"):
                        with st.expander(f"📚 Sources ({len(rag_result['sources'])} properties)"):
                            for i, source in enumerate(rag_result["sources"][:5], 1):
                                st.markdown(f"**Property {i}:**")
                                st.text(source["document"])
                                if source.get("metadata"):
                                    st.caption(f"Commune: {source['metadata'].get('commune', 'N/A')}, "
                                             f"Postal Code: {source['metadata'].get('postal_code', 'N/A')}")
                                st.divider()
                    
                    message_to_add = {
                        "role": "assistant",
                        "content": rag_result["answer"],
                        "mode": "rag",
                        "sources_count": len(rag_result.get("sources", [])),
                    }
            else:
                # Structured query mode
                with st.spinner("Analyzing your query..."):
                    response, result, params, sql_query, used_llm = process_query(
                        prompt, st.session_state.executor, st.session_state.parser
                    )
                    
                    # Show LLM usage indicator
                    if used_llm:
                        st.success("🤖 Parsed using LLM")
                    else:
                        st.info("📝 Parsed using regex (LLM unavailable or disabled)")
                    
                    st.markdown(response)

                    if result.get("success"):
                        with st.expander("📋 Query details"):
                            st.json(params)
                            if used_llm:
                                st.caption("✅ Parsed with LLM")
                            else:
                                st.caption("⚠️ Parsed with regex fallback")
                        
                        with st.expander("🗄️ SQL Query"):
                            st.code(sql_query, language="sql")

                message_to_add = {"role": "assistant", "content": response, "mode": "structured"}
                if result.get("success"):
                    message_to_add["query_details"] = params
                    message_to_add["sql_query"] = sql_query
                    message_to_add["used_llm"] = used_llm
            
            st.session_state.messages.append(message_to_add)
            st.rerun()


if __name__ == "__main__":
    main()
