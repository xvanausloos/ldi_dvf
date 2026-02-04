"""Tests for dvf.load."""

from pathlib import Path

import pandas as pd
import pytest

from dvf.load import load_dvf_raw


def test_load_dvf_raw_file_not_found() -> None:
    with pytest.raises(FileNotFoundError, match="not found"):
        load_dvf_raw(Path("/nonexistent/dvf.txt"))


def test_load_dvf_raw_csv_like(tmp_path: Path) -> None:
    """Load a pipe-separated sample file."""
    sample = tmp_path / "sample.txt"
    sample.write_text("id|date|nature\n1|2023-01-01|Vente\n2|2023-02-01|Vente", encoding="utf-8")
    df = load_dvf_raw(sample)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert list(df.columns) == ["id", "date", "nature"]
