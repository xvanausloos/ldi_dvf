"""Build the dataset of Marseille houses sold on a given date, with sale history.

Primary rows come from ONE file, `data/raw/ValeursFoncieres-2025.txt`: every
house (Maison) sold in Marseille on the target date (default 15/09/2025).

Each such house is then enriched with its `previous_mutations`: a JSON array of
the parcel's earlier sales (date + price), found by scanning *all* the raw DVF
year files in `data/raw/`. This gives, per house sold on the target date, the
full prior transaction history of its cadastral parcel.

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

USE_COLS = [
    "Date mutation",
    "Nature mutation",
    "Valeur fonciere",
    "No voie",
    "B/T/Q",
    "Type de voie",
    "Voie",
    "Code postal",
    "Commune",
    "Code departement",
    "Code commune",
    "Prefixe de section",
    "Section",
    "No plan",
    "Surface reelle bati",
    "Nombre pieces principales",
    "Surface terrain",
    "Code type local",
]


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
    """No plan as a bare integer string ('38')."""
    s = clean(x)
    if s.endswith(".0"):
        s = s[:-2]
    return str(int(s)) if s.isdigit() else s


def load_marseille_houses(path: Path) -> pd.DataFrame:
    """Marseille house (Maison) rows from one raw DVF file, keyed by parcel."""
    df = pd.read_csv(path, sep="|", encoding="utf-8", low_memory=False, dtype=str)
    df["Code commune"] = df["Code commune"].str.zfill(3)
    mask = (
        (df["Code departement"] == "13")
        & (df["Code commune"].isin(MARSEILLE_COMMUNE_CODES))
        & (df["Code type local"] == "1")  # Maison
    )
    sub = df.loc[mask, [c for c in USE_COLS if c in df.columns]].copy()
    sub["insee_code"] = "13" + sub["Code commune"]
    for c in KEY:
        sub[c] = sub[c].fillna("").astype(str).str.strip()
    return sub


def collapse_sale(group: pd.DataFrame) -> dict:
    """Collapse rows of one (date, price) sale (several lots) into one mutation."""
    first = group.iloc[0]
    bati = sum(v for v in (to_float(x) for x in group["Surface reelle bati"]) if v is not None)
    pieces = [to_float(x) for x in group["Nombre pieces principales"]]
    pieces = [p for p in pieces if p is not None]
    terrain = next(
        (v for v in (to_float(x) for x in group["Surface terrain"]) if v is not None), None
    )
    return {
        "date_mutation": to_iso(first["Date mutation"]),
        "nature_mutation": clean(first.get("Nature mutation")) or None,
        "valeur_fonciere": to_float(first["Valeur fonciere"]),
        "surface_reelle_bati": bati or None,
        "nombre_pieces_principales": max(pieces) if pieces else None,
        "surface_terrain": terrain,
        "nb_lignes": len(group),
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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--date", default="15/09/2025", help="Sale date dd/mm/YYYY (default 15/09/2025)")
    args = p.parse_args()

    target_iso = to_iso(args.date)
    if target_iso is None:
        raise SystemExit(f"Invalid --date {args.date!r}; expected dd/mm/YYYY")

    # 1. Houses sold on the target date, from the 2025 file only.
    primary = load_marseille_houses(RAW / PRIMARY_FILE)
    sold = primary[primary["Date mutation"].map(to_iso) == target_iso]
    target_keys = set(map(tuple, sold[KEY].values.tolist()))
    print(
        f"Marseille house rows sold on {args.date}: {len(sold)} across {len(target_keys)} parcels"
    )
    if not target_keys:
        raise SystemExit("No Marseille house sold on that date.")

    # 2. Every mutation of those parcels, across all raw year files.
    files = sorted(RAW.glob("ValeursFoncieres-*.txt"))
    frames = []
    for f in files:
        sub = load_marseille_houses(f)
        keep = sub[list(map(lambda r: tuple(r) in target_keys, sub[KEY].values.tolist()))]
        if len(keep):
            print(f"  {f.name}: {len(keep)} rows on target parcels")
            frames.append(keep)
    allrows = pd.concat(frames, ignore_index=True)

    # 3. One row per house: the target-date sale + JSON history of earlier sales.
    records = []
    for key, parcel in allrows.groupby(KEY, dropna=False, sort=False):
        muts = parcel_mutations(parcel)
        sale = next((m for m in muts if m["date_mutation"] == target_iso), None)
        if sale is None:
            continue
        previous = [
            {"date_mutation": m["date_mutation"], "valeur_fonciere": m["valeur_fonciere"]}
            for m in muts
            if m["date_mutation"] and m["date_mutation"] < target_iso
        ]
        # Address / identity from a target-date row.
        row = parcel[parcel["Date mutation"].map(to_iso) == target_iso].iloc[0]
        insee, prefixe, section, plan = key
        no_voie = norm_plan(row.get("No voie"))
        btq = clean(row.get("B/T/Q"))
        commune = clean(row.get("Commune"))
        street = " ".join(
            s for s in [no_voie, btq, clean(row.get("Type de voie")), clean(row.get("Voie"))] if s
        ).strip()
        cp = clean(row.get("Code postal"))
        records.append(
            {
                "insee_code": insee,
                "Commune": commune,
                "code_postal": cp,
                "adresse": f"{street}, {cp} {commune}".strip(),
                "prefixe_section": prefixe,
                "Section": section,
                "No plan": plan,
                "id_parcelle": f"{insee}{prefixe.zfill(3)}{section.zfill(2)}{norm_plan(plan).zfill(4)}",
                "No voie": no_voie,
                "btq": btq,
                "Type de voie": clean(row.get("Type de voie")),
                "Voie": clean(row.get("Voie")),
                "date_mutation": sale["date_mutation"],
                "nature_mutation": sale["nature_mutation"],
                "valeur_fonciere": sale["valeur_fonciere"],
                "surface_reelle_bati": sale["surface_reelle_bati"],
                "nombre_pieces_principales": sale["nombre_pieces_principales"],
                "surface_terrain": sale["surface_terrain"],
                "n_previous_mutations": len(previous),
                "previous_mutations": json.dumps(previous, ensure_ascii=False),
            }
        )

    out = pd.DataFrame(records).sort_values(["insee_code", "Section", "No plan"])
    PROC.mkdir(parents=True, exist_ok=True)
    out_path = PROC / f"marseille_houses_sold_{target_iso}.csv"
    out.to_csv(out_path, index=False)

    print("-" * 55)
    print(f"Houses sold on {args.date}: {len(out)}")
    print(f"With prior sale history: {(out['n_previous_mutations'] > 0).sum()}")
    print(f"Saved -> {out_path}")
    print("\n--- sample ---")
    s = out.iloc[out["n_previous_mutations"].values.argmax()]
    print(f"  {s['adresse']}  (id_parcelle {s['id_parcelle']})")
    print(
        f"  sold {s['date_mutation']} for {s['valeur_fonciere']:.0f} EUR, "
        f"{s['surface_reelle_bati']} m2"
    )
    print(f"  previous_mutations: {s['previous_mutations']}")


if __name__ == "__main__":
    main()
