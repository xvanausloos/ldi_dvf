"""Extract all DVF mutations for houses (Maison) in Marseille, all arrondissements.

Marseille is NOT coded as INSEE 13055 in DVF. Like Paris and Lyon (the "PLM"
cities), it is recorded at the arrondissement level: INSEE 13201..13216
(Code departement=13, Code commune=201..216). So the geographic filter is the
16 arrondissement codes, not 13055 (which returns zero rows).

Houses only => Code type local == 1 ("Maison").
"""

from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
OUT = Path(__file__).resolve().parents[1] / "data" / "processed" / "marseille_houses_dvf.csv"

FILES = [
    "ValeursFoncieres-2020-S2.txt",
    "ValeursFoncieres-2021.txt",
    "ValeursFoncieres-2022.txt",
    "ValeursFoncieres-2023.txt",
    "ValeursFoncieres-2024.txt",
    "ValeursFoncieres-2025-S1.txt",
]

# 13201..13216 -> commune codes 201..216 within department 13
MARSEILLE_COMMUNE_CODES = {f"{c:03d}" for c in range(201, 217)}

frames = []
for name in FILES:
    path = RAW / name
    df = pd.read_csv(path, sep="|", encoding="utf-8", low_memory=False, dtype=str)
    df["Code commune"] = df["Code commune"].str.zfill(3)
    mask = (
        (df["Code departement"] == "13")
        & (df["Code commune"].isin(MARSEILLE_COMMUNE_CODES))
        & (df["Code type local"] == "1")  # Maison
    )
    sub = df[mask].copy()
    sub["insee_code"] = "13" + sub["Code commune"]
    sub["source_file"] = name
    print(f"{name}: {len(sub):>6} house rows")
    frames.append(sub)

out = pd.concat(frames, ignore_index=True)
OUT.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(OUT, index=False)

print("-" * 50)
print(f"Total Marseille house rows: {len(out)}")
print("By arrondissement (insee_code):")
print(out.groupby(["insee_code", "Commune"]).size().to_string())
print(f"\nSaved -> {OUT}")
