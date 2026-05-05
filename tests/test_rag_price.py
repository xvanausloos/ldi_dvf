"""RAG document text includes mutation prices from string literals."""

import pandas as pd

from dvf.rag import DVFVectorStore, _parse_price_raw


def test_parse_price_raw_european_string():
    assert _parse_price_raw("120000,00") == 120000.0
    assert _parse_price_raw("625000,50") == 625000.50


def test_create_text_representation_includes_string_mutation_price():
    store = object.__new__(DVFVectorStore)
    row = pd.Series(
        {
            "Commune": "ENSUES-LA-REDONNE",
            "Code postal": 13820.0,
            "Type local": "Maison",
            "Surface reelle bati": 90.0,
            "Nombre pieces principales": 4.0,
            "Voie": "RUE TEST",
            "mutations": "[{'22/07/2021': '120000,00'}]",
        }
    )
    text = DVFVectorStore._create_text_representation(store, row)
    assert "120000" in text
    assert "22/07/2021" in text
    assert "dernier prix" in text.lower()


def test_create_text_representation_picks_latest_date_across_blocks():
    store = object.__new__(DVFVectorStore)
    row = pd.Series(
        {
            "Commune": "X",
            "Code postal": 13820.0,
            "Type local": "Maison",
            "Surface reelle bati": 80.0,
            "Nombre pieces principales": 3.0,
            "Voie": None,
            "mutations": (
                "[{'01/01/2020': '100000,00'}, {'01/06/2022': '250000,00'}]"
            ),
        }
    )
    text = DVFVectorStore._create_text_representation(store, row)
    assert "250000" in text
    assert "01/06/2022" in text
