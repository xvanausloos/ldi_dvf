"""Streamlit chat interface for querying DVF real estate data."""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from dvf.query import QueryExecutor, QueryParser

DATA_PATH = Path(__file__).parent / "data" / "processed" / "df_grouped_2020_2025_france_cleaned.csv"


@st.cache_data
def load_data():
    """Load and prepare DVF data."""
    if not DATA_PATH.exists():
        st.error(f"Data file not found: {DATA_PATH}")
        st.stop()
    df = pd.read_csv(DATA_PATH, low_memory=False)
    return df


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
) -> tuple[str, dict, dict]:
    """Process a query and return the formatted response, result dict, and params dict."""
    params = parser.parse(prompt)
    result = executor.execute(params)
    return format_result(result), result, params


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
            st.session_state.parser = QueryParser()

    with st.sidebar:
        st.header("⚙️ Controls")
        if st.button("🗑️ Clear Chat History", use_container_width=True, type="secondary"):
            st.session_state.messages = []
            st.session_state.pending_query = None
            st.rerun()

        st.divider()

        st.header("💡 Example Queries")
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
        st.metric("Total Properties", f"{len(st.session_state.df):,}")
        st.metric(
            "With Price Data",
            f"{st.session_state.df['mutations'].notna().sum():,}",
        )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "query_details" in message:
                with st.expander("Query details"):
                    st.json(message["query_details"])

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
            with st.spinner("Analyzing your query..."):
                response, result, params = process_query(
                    prompt, st.session_state.executor, st.session_state.parser
                )
                st.markdown(response)

                if result.get("success"):
                    with st.expander("Query details"):
                        st.json(params)

        message_to_add = {"role": "assistant", "content": response}
        if result.get("success"):
            message_to_add["query_details"] = params
        st.session_state.messages.append(message_to_add)
        st.rerun()


if __name__ == "__main__":
    main()
