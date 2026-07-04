"""Tests for postal code matching on grouped DVF-like DataFrames."""

import pandas as pd

from dvf.query import QueryExecutor, match_postal_code_column


def test_match_postal_code_column_float_vs_string():
    s = pd.Series([13008.0, 75001.0, 13008])
    mask = match_postal_code_column(s, "13008")
    assert mask.tolist() == [True, False, True]


def test_executor_filters_float_postal_like_processed_csv():
    df = pd.DataFrame(
        {
            "Code postal": [13008.0],
            "Commune": ["MARSEILLE"],
            "Type local": ["Maison"],
            "Surface reelle bati": [100.0],
            "mutations": ["[{'15/06/2024': '450000'}]"],
        }
    )
    ex = QueryExecutor(df)
    result = ex.execute(
        {"postal_code": "13008", "commune": None, "query_type": "mean"}
    )
    assert result["success"] is True
    assert result["count"] == 1


def test_executor_no_rows_when_postal_mismatch():
    df = pd.DataFrame(
        {
            "Code postal": [13008.0],
            "Commune": ["MARSEILLE"],
            "Type local": ["Maison"],
            "Surface reelle bati": [80.0],
            "mutations": ["[{'15/06/2024': '300000'}]"],
        }
    )
    ex = QueryExecutor(df)
    result = ex.execute(
        {"postal_code": "75001", "commune": None, "query_type": "count"}
    )
    assert result["success"] is False
    assert result["count"] == 0
