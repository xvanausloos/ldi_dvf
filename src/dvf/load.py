"""Load DVF data from official formats (pipe-separated .txt or CSV)."""

from pathlib import Path

import pandas as pd


# DVF raw format: pipe-separated .txt (official millésimes from data.gouv.fr)
# Columns may vary by year; common ones: id_mutation, date_mutation, nature_mutation, etc.
DEFAULT_DVF_SEP = "|"
DEFAULT_ENCODING = "utf-8"


def load_dvf_raw(
    path: str | Path,
    *,
    sep: str = DEFAULT_DVF_SEP,
    encoding: str = DEFAULT_ENCODING,
    nrows: int | None = None,
    low_memory: bool = True,
) -> pd.DataFrame:
    """Load DVF data from official pipe-separated .txt file.

    Official DVF files use pipe (|) as separator. Large files: use nrows for
    testing or chunking for full load.

    Args:
        path: Path to .txt or .csv file.
        sep: Column separator (default pipe for official DVF).
        encoding: File encoding.
        nrows: Limit number of rows (useful for large files).
        low_memory: Use chunked reading for memory efficiency.

    Returns:
        DataFrame with DVF mutations.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"DVF file not found: {path}")

    return pd.read_csv(
        path,
        sep=sep,
        encoding=encoding,
        nrows=nrows,
        low_memory=low_memory,
        on_bad_lines="warn",
    )


def load_dvf_csv(
    path: str | Path,
    *,
    encoding: str = DEFAULT_ENCODING,
    nrows: int | None = None,
) -> pd.DataFrame:
    """Load DVF data from comma-separated CSV.

    For DVF+ files from data.gouv.fr (often pipe-separated even with .csv extension),
    use load_dvf_plus() or load_dvf_raw(path, sep="|") instead.
    """
    return load_dvf_raw(path, sep=",", encoding=encoding, nrows=nrows, low_memory=True)


def load_dvf_plus(
    path: str | Path,
    *,
    sep: str | None = None,
    encoding: str = DEFAULT_ENCODING,
    nrows: int | None = None,
    low_memory: bool = True,
) -> pd.DataFrame:
    """Load DVF+ or DVF-style mutation data.

    By default uses pipe (|) as separator (e.g. dvf_plus.csv from data.gouv.fr).
    For CSV with semicolon (;) use sep=";", e.g. mutations_d13.csv.

    Args:
        path: Path to the file.
        sep: Column separator. Default "|"; use ";" for semicolon-separated CSV.
        encoding: File encoding.
        nrows: Limit number of rows (useful for large files).
        low_memory: Use chunked reading for memory efficiency.

    Returns:
        DataFrame with DVF mutations.
    """
    if sep is None:
        sep = DEFAULT_DVF_SEP
    return load_dvf_raw(
        path,
        sep=sep,
        encoding=encoding,
        nrows=nrows,
        low_memory=low_memory,
    )


def get_data_dir() -> Path:
    """Return project data directory (raw by default)."""
    return Path(__file__).resolve().parents[2] / "data" / "raw"
