"""Analysis helpers for DVF mutation data."""

import pandas as pd


def summarize_mutations(df: pd.DataFrame) -> pd.DataFrame:
    """Basic summary of mutations: count by type and (if present) by date.

    Expects columns such as nature_mutation, date_mutation (or similar).
    Adapt column names to your DVF schema.
    """
    summary = {}
    if "nature_mutation" in df.columns:
        summary["by_nature"] = df["nature_mutation"].value_counts()
    if "date_mutation" in df.columns:
        df_date = df.copy()
        df_date["date_mutation"] = pd.to_datetime(df_date["date_mutation"], errors="coerce")
        df_date = df_date.dropna(subset=["date_mutation"])
        if not df_date.empty:
            summary["by_year"] = df_date["date_mutation"].dt.year.value_counts().sort_index()
    return pd.DataFrame(summary) if summary else pd.DataFrame()


def price_stats(df: pd.DataFrame, value_col: str = "valeur_fonciere") -> pd.Series:
    """Compute basic price statistics (if value column exists)."""
    if value_col not in df.columns:
        return pd.Series(dtype=float)
    s = pd.to_numeric(df[value_col], errors="coerce").dropna()
    return s.describe()
