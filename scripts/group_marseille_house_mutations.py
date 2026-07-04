"""Group Marseille house DVF rows into one row per house, mutations as JSON.

Input : data/processed/marseille_houses_dvf.csv  (houses only, all arrondissements)
Output: data/processed/marseille_houses_grouped.csv

"Same house" = unique cadastral parcel:
    insee_code + Prefixe de section + Section + No plan
(insee_code = Code departement + Code commune, e.g. 13211). This is more robust
than Code postal + Section + No plan because it cannot collide across
arrondissements and honours the section prefix.

Each house row carries a `mutations` column: a JSON array of the sale events on
that parcel. Multiple raw rows of the same sale (several lots) are collapsed into
one mutation, keyed by (date_mutation, valeur_fonciere).
"""

import json
from pathlib import Path

import pandas as pd

PROC = Path(__file__).resolve().parents[1] / "data" / "processed"
IN = PROC / "marseille_houses_dvf.csv"
OUT = PROC / "marseille_houses_grouped.csv"

df = pd.read_csv(IN, dtype=str, low_memory=False)


def to_float(x):
    """French decimal comma -> float; empty -> None."""
    if x is None or (isinstance(x, float)) or str(x).strip() == "" or str(x) == "nan":
        return None
    try:
        return float(str(x).replace(",", ".").replace("\xa0", "").strip())
    except ValueError:
        return None


def to_iso(d):
    """DVF date dd/mm/YYYY -> YYYY-MM-DD."""
    d = str(d).strip()
    if not d or d == "nan":
        return None
    try:
        return pd.to_datetime(d, format="%d/%m/%Y").date().isoformat()
    except ValueError:
        return None


# Parcel (house) key. Fill NA so groupby keeps rows with empty prefix/section.
KEY = ["insee_code", "Prefixe de section", "Section", "No plan"]
for c in KEY:
    df[c] = df[c].fillna("").astype(str).str.strip()

# House-level attributes copied from the first row of each group.
ATTR = [
    "insee_code", "Commune", "Code postal", "Code departement", "Code commune",
    "Prefixe de section", "Section", "No plan", "No voie", "B/T/Q",
    "Type de voie", "Voie",
]


def build_mutations(group: pd.DataFrame) -> tuple[str, int]:
    """Collapse rows into a JSON list of mutations, one per (date, price) sale."""
    muts = {}
    for _, r in group.iterrows():
        date = to_iso(r["Date mutation"])
        price = to_float(r["Valeur fonciere"])
        k = (date, price)
        m = muts.get(k)
        bati = to_float(r["Surface reelle bati"])
        pieces = to_float(r["Nombre pieces principales"])
        terrain = to_float(r["Surface terrain"])
        if m is None:
            muts[k] = {
                "date_mutation": date,
                "nature_mutation": (r.get("Nature mutation") or "").strip() or None,
                "valeur_fonciere": price,
                "surface_reelle_bati": bati,
                "nombre_pieces_principales": pieces,
                "surface_terrain": terrain,
                "nb_lignes": 1,
            }
        else:
            # Same sale spanning several lot rows: aggregate the physical detail.
            m["nb_lignes"] += 1
            if bati is not None:
                m["surface_reelle_bati"] = (m["surface_reelle_bati"] or 0) + bati
            if pieces is not None:
                m["nombre_pieces_principales"] = max(
                    m["nombre_pieces_principales"] or 0, pieces
                )
            if terrain is not None and m["surface_terrain"] is None:
                m["surface_terrain"] = terrain
    ordered = sorted(
        muts.values(), key=lambda x: (x["date_mutation"] or "", x["valeur_fonciere"] or 0)
    )
    return json.dumps(ordered, ensure_ascii=False), len(ordered)


rows = []
groups = df.groupby(KEY, dropna=False, sort=False)
n = len(groups)
print(f"Grouping {len(df)} house rows into {n} unique parcels...")
for i, (_, g) in enumerate(groups, start=1):
    info = {a: g.iloc[0].get(a) for a in ATTR}
    mut_json, n_mut = build_mutations(g)
    info["n_mutations"] = n_mut
    info["mutations"] = mut_json
    rows.append(info)
    if i % 2000 == 0 or i == n:
        print(f"  {i}/{n} parcels")

out = pd.DataFrame(rows).sort_values(["insee_code", "Section", "No plan"])
out.to_csv(OUT, index=False)

print("-" * 55)
print(f"Unique houses (parcels): {len(out)}")
print(f"Total mutations across all houses: {out['n_mutations'].astype(int).sum()}")
print("Houses with >1 mutation (resold in period): "
      f"{(out['n_mutations'].astype(int) > 1).sum()}")
print("\nMutations-per-house distribution:")
print(out["n_mutations"].astype(int).value_counts().sort_index().to_string())
print(f"\nSaved -> {OUT}")
print("\n--- sample row ---")
s = out.iloc[(out['n_mutations'].astype(int)).values.argmax()]
print(f"{s['insee_code']} {s['Commune']} | Section {s['Section']} plan {s['No plan']} "
      f"| {s['No voie']} {s['Voie']}")
print(json.dumps(json.loads(s["mutations"]), ensure_ascii=False, indent=2))
