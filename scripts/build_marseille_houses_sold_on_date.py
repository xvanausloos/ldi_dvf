"""Build the dataset of Marseille houses sold on a given date, with sale history.

Primary rows come from ONE file, `data/raw/ValeursFoncieres-2025.txt`: every
house (Maison) sold in Marseille on the target date (default 15/09/2025).

Every raw DVF column of that sale is carried through verbatim (nothing dropped),
and the following derived columns are appended:

    insee_code           13 + Code commune (e.g. 13207)
    adresse              ready-to-paste "<no> <btq> <type> <voie>, <cp> <commune>"
    id_parcelle          14-char cadastral id = code_commune(5)+prefixe(3)+section(2)+plan(4)
    n_previous_mutations count of earlier sales of the same parcel
    previous_mutations   JSON array of earlier sales: date, nature, price, surface

The history is found by scanning *all* the raw DVF year files in `data/raw/`.

Marseille is NOT INSEE 13055 in DVF. Like Paris and Lyon it is recorded per
arrondissement: Code departement=13, Code commune=201..216 (INSEE 13201..13216).

    uv run python scripts/build_marseille_houses_sold_on_date.py
    uv run python scripts/build_marseille_houses_sold_on_date.py --date 15/09/2025

Output: data/processed/marseille_houses_sold_<YYYY-MM-DD>.csv
"""

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"

# The single file that defines who was sold on the target date.
PRIMARY_FILE = "ValeursFoncieres-2025.txt"

# 13201..13216 -> commune codes 201..216 within department 13.
MARSEILLE_COMMUNE_CODES = {f"{c:03d}" for c in range(201, 217)}

# Cadastral parcel identity = "same house".
KEY = ["insee_code", "Prefixe de section", "Section", "No plan"]

# Derived columns appended after the raw ones (kept out of the "raw" block).
DERIVED = ["insee_code", "adresse", "id_parcelle", "n_previous_mutations", "previous_mutations"]


def to_float(x):
    """French decimal comma -> float; empty -> None."""
    s = str(x).strip()
    if s in ("", "nan"):
        return None
    try:
        return float(s.replace(",", ".").replace("\xa0", ""))
    except ValueError:
        return None


def to_iso(d):
    """DVF date dd/mm/YYYY -> YYYY-MM-DD (or None)."""
    s = str(d).strip()
    if not s or s == "nan":
        return None
    try:
        return pd.to_datetime(s, format="%d/%m/%Y").date().isoformat()
    except ValueError:
        return None


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
    """All raw columns of Marseille house (Maison) rows, plus insee_code + key."""
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


def collapse_sale(group: pd.DataFrame) -> dict:
    """Collapse rows of one (date, price) sale (several lots) into one mutation."""
    first = group.iloc[0]
    bati = sum(v for v in (to_float(x) for x in group["Surface reelle bati"]) if v is not None)
    return {
        "date_mutation": to_iso(first["Date mutation"]),
        "nature_mutation": clean(first.get("Nature mutation")) or None,
        "valeur_fonciere": to_float(first["Valeur fonciere"]),
        "surface_reelle_bati": bati or None,
    }


def parcel_mutations(parcel: pd.DataFrame) -> list[dict]:
    """All distinct sales on a parcel, chronological, collapsing multi-lot rows."""
    parcel = parcel.copy()
    parcel["_date"] = parcel["Date mutation"].map(to_iso)
    parcel["_price"] = parcel["Valeur fonciere"].map(to_float)
    muts = [
        collapse_sale(g) for _, g in parcel.groupby(["_date", "_price"], dropna=False, sort=False)
    ]
    return sorted(muts, key=lambda m: (m["date_mutation"] or "", m["valeur_fonciere"] or 0))


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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--date", default="15/09/2025", help="Sale date dd/mm/YYYY (default 15/09/2025)")
    args = p.parse_args()

    target_iso = to_iso(args.date)
    if target_iso is None:
        raise SystemExit(f"Invalid --date {args.date!r}; expected dd/mm/YYYY")

    # 1. Houses sold on the target date, from the 2025 file only. Keep raw column order.
    primary = load_marseille_houses(RAW / PRIMARY_FILE)
    raw_cols = [c for c in primary.columns if c != "insee_code"]
    sold = primary[primary["Date mutation"].map(to_iso) == target_iso]
    target_keys = set(map(tuple, sold[KEY].values.tolist()))
    print(
        f"Marseille house rows sold on {args.date}: {len(sold)} across {len(target_keys)} parcels"
    )
    if not target_keys:
        raise SystemExit("No Marseille house sold on that date.")

    # 2. Every mutation of those parcels, across all raw year files (for history).
    frames = []
    for f in sorted(RAW.glob("ValeursFoncieres-*.txt")):
        sub = load_marseille_houses(f)
        keep = sub[[tuple(r) in target_keys for r in sub[KEY].values.tolist()]]
        if len(keep):
            print(f"  {f.name}: {len(keep)} rows on target parcels")
            frames.append(keep)
    history = pd.concat(frames, ignore_index=True)

    # 3. One row per house: the full raw target-date sale + JSON history of earlier sales.
    hist_by_key = {
        key: parcel_mutations(g) for key, g in history.groupby(KEY, dropna=False, sort=False)
    }

    records = []
    for key, sale_rows in sold.groupby(KEY, dropna=False, sort=False):
        if len(sale_rows) > 1:
            print(
                f"  NOTE: parcel {key} has {len(sale_rows)} raw rows on {args.date}; "
                "keeping the first (multi-lot sale)."
            )
        row = sale_rows.iloc[0]
        previous = [
            {
                "date_mutation": m["date_mutation"],
                "nature_mutation": m["nature_mutation"],
                "valeur_fonciere": m["valeur_fonciere"],
                "surface_reelle_bati": m["surface_reelle_bati"],
            }
            for m in hist_by_key.get(key, [])
            if m["date_mutation"] and m["date_mutation"] < target_iso
        ]
        rec = {c: row[c] for c in raw_cols}
        rec["insee_code"] = row["insee_code"]
        rec["adresse"] = build_adresse(row)
        rec["id_parcelle"] = build_id_parcelle(row)
        rec["n_previous_mutations"] = len(previous)
        rec["previous_mutations"] = json.dumps(previous, ensure_ascii=False)
        records.append(rec)

    out = pd.DataFrame(records, columns=raw_cols + DERIVED).sort_values(
        ["insee_code", "Section", "No plan"]
    )
    PROC.mkdir(parents=True, exist_ok=True)
    out_path = PROC / f"marseille_houses_sold_{target_iso}.csv"
    out.to_csv(out_path, index=False)

    print("-" * 55)
    print(
        f"Houses sold on {args.date}: {len(out)}  ({len(out.columns)} columns, all raw + derived)"
    )
    print(f"With prior sale history: {(out['n_previous_mutations'] > 0).sum()}")
    print(f"Saved -> {out_path}")
    print("\n--- sample (house with history) ---")
    s = out.iloc[out["n_previous_mutations"].values.argmax()]
    print(f"  {s['adresse']}  (id_parcelle {s['id_parcelle']})")
    print(f"  sold {s['Date mutation']} for {s['Valeur fonciere']} EUR")
    print(f"  previous_mutations: {s['previous_mutations']}")


if __name__ == "__main__":
    main()
