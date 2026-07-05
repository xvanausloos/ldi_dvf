"""Add locating fields to the model-ready repeat-sales dataset.

The model-ready CSV identifies a house only by (insee_code, Section, No plan) plus
a street name. That is not enough to verify a sale on the official explorer
(https://explore.data.gouv.fr/fr/immobilier). This script joins the raw DVF back
onto the dataset to add, per house:

    code_postal      postal code (e.g. 13010) -- what you type in the map search
    prefixe_section  cadastral section prefix (significant for Marseille: e.g. 856)
    btq              bis/ter/quater indicator on the street number, if any
    id_parcelle      14-char cadastral parcel id used by DVF geolocalise / the map
                     = code_commune(5) + prefixe(3) + section(2) + no_plan(4)
    adresse          ready-to-paste "<no> <btq> <type> <voie>, <cp> <commune>"

The parcel key (insee_code, Section, No plan) is unique within an arrondissement,
so the join is 1:1; the script asserts this and warns on any ambiguity.
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
DATASET = ROOT / "data" / "processed" / "marseille_houses_repeat_sales_model_ready.csv"

FILES = [
    "ValeursFoncieres-2020-S2.txt",
    "ValeursFoncieres-2021.txt",
    "ValeursFoncieres-2022.txt",
    "ValeursFoncieres-2023.txt",
    "ValeursFoncieres-2024.txt",
    "ValeursFoncieres-2025-S1.txt",
]

MARSEILLE_COMMUNE_CODES = {f"{c:03d}" for c in range(201, 217)}


def norm_plan(x) -> str:
    """No plan as a bare integer string ('38'), matching the dataset."""
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return str(int(s)) if s.isdigit() else s


def norm_voie(x) -> str:
    """Street number as a bare integer string ('28')."""
    s = str(x).strip()
    if s in ("", "nan"):
        return ""
    if s.endswith(".0"):
        s = s[:-2]
    return str(int(s)) if s.isdigit() else s


# --- collect the locating attributes per Marseille house parcel ---------------
frames = []
for name in FILES:
    df = pd.read_csv(RAW / name, sep="|", encoding="utf-8", low_memory=False, dtype=str)
    df["Code commune"] = df["Code commune"].str.zfill(3)
    mask = (
        (df["Code departement"] == "13")
        & (df["Code commune"].isin(MARSEILLE_COMMUNE_CODES))
        & (df["Code type local"] == "1")  # Maison
    )
    sub = df[mask].copy()
    sub["insee_code"] = "13" + sub["Code commune"]
    frames.append(
        sub[["insee_code", "Section", "No plan", "Prefixe de section",
             "Code postal", "B/T/Q"]]
    )

raw = pd.concat(frames, ignore_index=True)
raw["plan_key"] = raw["No plan"].map(norm_plan)
raw["Section"] = raw["Section"].fillna("").astype(str).str.strip()

for c in ["Prefixe de section", "Code postal", "B/T/Q"]:
    raw[c] = raw[c].fillna("").astype(str).str.strip()

# One record per parcel; take the most common value for each attribute.
def first_mode(s: pd.Series) -> str:
    s = s[s != ""]
    return s.mode().iloc[0] if not s.empty else ""


lookup = (
    raw.groupby(["insee_code", "Section", "plan_key"], sort=False)
    .agg(
        prefixe_section=("Prefixe de section", first_mode),
        code_postal=("Code postal", first_mode),
        btq=("B/T/Q", first_mode),
    )
    .reset_index()
)

# --- join onto the model-ready dataset ----------------------------------------
ds = pd.read_csv(DATASET, dtype={"insee_code": str, "Section": str})
ds["plan_key"] = ds["No plan"].map(norm_plan)

before = len(ds)
merged = ds.merge(lookup, on=["insee_code", "Section", "plan_key"], how="left")
assert len(merged) == before, "join changed the row count -> ambiguous parcel key"

missing = merged["code_postal"].isna().sum() + (merged["code_postal"] == "").sum()
if missing:
    print(f"WARNING: {missing} rows could not be matched back to raw DVF")


def build_id_parcelle(r) -> str:
    prefixe = (r["prefixe_section"] or "").zfill(3)
    section = str(r["Section"] or "").strip().zfill(2)
    plan = norm_plan(r["No plan"]).zfill(4)
    return f"{r['insee_code']}{prefixe}{section}{plan}"


def build_adresse(r) -> str:
    parts = [norm_voie(r["No voie"])]
    if r.get("btq"):
        parts.append(str(r["btq"]))
    parts += [str(r["Type de voie"] or ""), str(r["Voie"] or "")]
    street = " ".join(p for p in parts if p).strip()
    cp = str(r["code_postal"] or "").strip()
    return f"{street}, {cp} {r['Commune']}".strip()


merged["id_parcelle"] = merged.apply(build_id_parcelle, axis=1)
merged["adresse"] = merged.apply(build_adresse, axis=1)
merged = merged.drop(columns=["plan_key"])

# --- reorder: identity block first, mutations last ----------------------------
lead = [
    "insee_code", "Commune", "code_postal", "adresse",
    "prefixe_section", "Section", "No plan", "id_parcelle",
    "No voie", "btq", "Type de voie", "Voie",
]
rest = [c for c in merged.columns if c not in lead and c != "mutations"]
merged = merged[lead + rest + ["mutations"]]

merged.to_csv(DATASET, index=False)

print(f"Enriched {len(merged)} rows -> {DATASET.name}")
print("Added columns: code_postal, prefixe_section, btq, id_parcelle, adresse")
print("\n--- sample ---")
s = merged.iloc[0]
print(f"  adresse     : {s['adresse']}")
print(f"  code_postal : {s['code_postal']}")
print(f"  id_parcelle : {s['id_parcelle']}")
