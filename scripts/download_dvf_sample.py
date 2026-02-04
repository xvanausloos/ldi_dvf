#!/usr/bin/env -S uv run
"""Download a DVF sample or list available datasets from data.gouv.fr.

DVF open data: https://www.data.gouv.fr/fr/datasets/demandes-de-valeurs-foncieres/
DVF+ (enriched): https://www.data.gouv.fr/fr/datasets/dvf-open-data/

Place downloaded files in data/raw/ then load with dvf.load.load_dvf_raw().
"""

from pathlib import Path

# Example: use requests to download from data.gouv.fr API
# import requests

# Example: DVF+ API or direct file URL – replace with actual resource URL from data.gouv.fr
# DVF millésimes are large; for a small sample you may use a subset or DVF+ CSV export.
DATA_RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
DATA_RAW.mkdir(parents=True, exist_ok=True)


def main() -> None:
    print(f"Data directory: {DATA_RAW}")
    print(
        "To get DVF data:\n"
        "  1. Go to https://www.data.gouv.fr/fr/datasets/demandes-de-valeurs-foncieres/\n"
        "  2. Or DVF+ https://www.data.gouv.fr/fr/datasets/dvf-open-data/\n"
        "  3. Download the desired millésime (pipe-separated .txt or CSV)\n"
        "  4. Save files into data/raw/"
    )
    # Optional: uncomment and set a real sample URL to auto-download a small file
    # url = "https://..."
    # r = requests.get(url, timeout=60)
    # out = DATA_RAW / "dvf_sample.csv"
    # out.write_bytes(r.content)
    # print(f"Downloaded: {out}")


if __name__ == "__main__":
    main()
