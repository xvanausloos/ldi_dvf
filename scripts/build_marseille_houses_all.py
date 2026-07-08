"""Build the dataset of ALL Marseille houses (Maison) sold across every raw DVF year file.

One row per raw DVF house sale in Marseille, found by scanning every
`data/raw/ValeursFoncieres-*.txt`. All 43 raw DVF columns are carried through
verbatim (nothing dropped), and the following derived columns are appended:

    insee_code   13 + Code commune (e.g. 13207)
    adresse      ready-to-paste "<no> <btq> <type> <voie>, <cp> <commune>"
    id_parcelle  14-char cadastral id = code_commune(5)+prefixe(3)+section(2)+plan(4)

Marseille is NOT INSEE 13055 in DVF. Like Paris and Lyon it is recorded per
arrondissement: Code departement=13, Code commune=201..216 (INSEE 13201..13216).

    uv run python scripts/build_marseille_houses_all.py

Output: data/processed/marseille_houses_all.csv
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"

# 13201..13216 -> commune codes 201..216 within department 13.
MARSEILLE_COMMUNE_CODES = {f"{c:03d}" for c in range(201, 217)}

# Derived columns appended after the raw ones (kept out of the "raw" block).
DERIVED = ["insee_code", "adresse", "id_parcelle"]


def clean(x) -> str:
    """Trimmed string, with NaN/None/'nan' collapsed to ''."""
    if x is None:
        return ""
    s = str(x).strip()
    return "" if s.lower() == "nan" else s


def norm_plan(x) -> str:
    """No plan / No voie as a bare integer string ('38')."""
    s = clean(x)
    if s.endswith(".0"):
        s = s[:-2]
    return str(int(s)) if s.isdigit() else s


def load_marseille_houses(path: Path) -> pd.DataFrame:
    """All raw columns of Marseille house (Maison) rows, plus insee_code."""
    df = pd.read_csv(path, sep="|", encoding="utf-8", low_memory=False, dtype=str)
    cc = df["Code commune"].str.zfill(3)
    mask = (
        (df["Code departement"] == "13")
        & (cc.isin(MARSEILLE_COMMUNE_CODES))
        & (df["Code type local"] == "1")  # Maison
    )
    sub = df[mask].copy()
    sub["insee_code"] = "13" + cc[mask]
    for c in ["Prefixe de section", "Section", "No plan"]:
        sub[c] = sub[c].fillna("").astype(str).str.strip()
    return sub


def build_adresse(row: pd.Series) -> str:
    parts = [
        norm_plan(row.get("No voie")),
        clean(row.get("B/T/Q")),
        clean(row.get("Type de voie")),
        clean(row.get("Voie")),
    ]
    street = " ".join(p for p in parts if p).strip()
    cp = clean(row.get("Code postal"))
    return f"{street}, {cp} {clean(row.get('Commune'))}".strip()


def build_id_parcelle(row: pd.Series) -> str:
    return (
        f"{clean(row['insee_code'])}"
        f"{clean(row['Prefixe de section']).zfill(3)}"
        f"{clean(row['Section']).zfill(2)}"
        f"{norm_plan(row['No plan']).zfill(4)}"
    )


def main() -> None:
    frames = []
    for f in sorted(RAW.glob("ValeursFoncieres-*.txt")):
        sub = load_marseille_houses(f)
        print(f"  {f.name}: {len(sub)} Marseille house rows")
        if len(sub):
            frames.append(sub)
    if not frames:
        raise SystemExit("No Marseille house rows found in data/raw/.")

    houses = pd.concat(frames, ignore_index=True)
    raw_cols = [c for c in houses.columns if c != "insee_code"]

    houses["adresse"] = houses.apply(build_adresse, axis=1)
    houses["id_parcelle"] = houses.apply(build_id_parcelle, axis=1)

    out = houses[raw_cols + DERIVED].sort_values(
        ["Date mutation", "insee_code", "Section", "No plan"]
    )
    PROC.mkdir(parents=True, exist_ok=True)
    out_path = PROC / "marseille_houses_all.csv"
    out.to_csv(out_path, index=False)

    print("-" * 55)
    print(f"All Marseille house sales: {len(out)} rows  ({len(out.columns)} columns, all raw + derived)")
    print(f"Distinct parcels: {out['id_parcelle'].nunique()}")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
