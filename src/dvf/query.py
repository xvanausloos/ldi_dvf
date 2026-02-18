"""Query parser and executor for natural language DVF data queries."""

import ast
import re
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd


def parse_mutations(mutations_str: str) -> list[tuple[datetime, float]]:
    """Parse mutations string into list of (date, price) tuples."""
    if pd.isna(mutations_str) or not mutations_str:
        return []
    try:
        mutations = ast.literal_eval(mutations_str)
        results = []
        for m in mutations:
            if isinstance(m, dict):
                for date_str, price_str in m.items():
                    try:
                        date = datetime.strptime(date_str, "%d/%m/%Y")
                        price = float(price_str.replace(",", "."))
                        results.append((date, price))
                    except (ValueError, AttributeError):
                        continue
        return sorted(results, key=lambda x: x[0])
    except (ValueError, SyntaxError, AttributeError):
        return []


def extract_postal_code(text: str) -> str | None:
    """Extract French postal code (5 digits) from text."""
    pattern = r"\b(\d{5})\b"
    matches = re.findall(pattern, text)
    if matches:
        return matches[0]
    return None


def extract_surface(text: str) -> float | None:
    """Extract surface area in m² from text."""
    patterns = [
        r"(\d+(?:[.,]\d+)?)\s*m[²2]",
        r"(\d+(?:[.,]\d+)?)\s*m2",
        r"(\d+(?:[.,]\d+)?)\s*square\s*meters",
        r"(\d+(?:[.,]\d+)?)\s*m\s*x\s*\d+",  # e.g., "100m x 50"
        r"(\d+(?:[.,]\d+)?)\s*m\b",  # e.g., "100m" (standalone)
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return float(match.group(1).replace(",", "."))
    return None


def extract_commune(text: str) -> str | None:
    """Extract commune name from text (case-insensitive)."""
    text_lower = text.lower()
    common_communes = [
        "ensues",
        "ensues la redonne",
        "marseille",
        "paris",
        "lyon",
        "toulouse",
        "nice",
        "nantes",
        "strasbourg",
        "montpellier",
        "bordeaux",
        "lille",
    ]
    for commune in common_communes:
        if commune in text_lower:
            if commune == "ensues":
                return "Ensues"
            return commune.title()
    return None


def extract_query_type(text: str) -> str:
    """Extract query type (mean, median, count, etc.) from text."""
    text_lower = text.lower()
    if any(word in text_lower for word in ["mean", "average", "moyenne", "moyen"]):
        return "mean"
    if any(word in text_lower for word in ["median", "mediane"]):
        return "median"
    if any(word in text_lower for word in ["count", "number", "nombre", "combien"]):
        return "count"
    if any(word in text_lower for word in ["min", "minimum", "minimum"]):
        return "min"
    if any(word in text_lower for word in ["max", "maximum", "maximum"]):
        return "max"
    return "mean"


class QueryParser:
    """Parse natural language queries about DVF data."""

    def __init__(self):
        self.postal_code: str | None = None
        self.commune: str | None = None
        self.surface_min: float | None = None
        self.surface_max: float | None = None
        self.query_type: str = "mean"

    def parse(self, query: str) -> dict[str, Any]:
        """Parse a natural language query and extract parameters."""
        query_lower = query.lower()

        postal_code = extract_postal_code(query)
        if postal_code:
            self.postal_code = postal_code

        commune = extract_commune(query)
        if commune:
            self.commune = commune

        surface = extract_surface(query)
        if surface:
            self.surface_min = surface * 0.9
            self.surface_max = surface * 1.1

        self.query_type = extract_query_type(query)

        return {
            "postal_code": self.postal_code,
            "commune": self.commune,
            "surface_min": self.surface_min,
            "surface_max": self.surface_max,
            "query_type": self.query_type,
        }


class QueryExecutor:
    """Execute queries on DVF DataFrame."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        if "parsed_mutations" not in self.df.columns:
            self.df["parsed_mutations"] = self.df["mutations"].apply(parse_mutations)
        if "last_price" not in self.df.columns:
            self.df["last_price"] = self.df["parsed_mutations"].apply(
                lambda x: x[-1][1] if x else np.nan
            )
        if "price_per_m2" not in self.df.columns:
            self.df["price_per_m2"] = (
                self.df["last_price"] / self.df["Surface reelle bati"]
            ).replace([np.inf, -np.inf], np.nan)

    def execute(self, params: dict[str, Any]) -> dict[str, Any]:
        """Execute query with given parameters."""
        filtered = self.df.copy()

        if params.get("postal_code"):
            filtered = filtered[
                filtered["Code postal"].astype(str) == str(params["postal_code"])
            ]

        if params.get("commune"):
            filtered = filtered[
                filtered["Commune"].str.contains(
                    params["commune"], case=False, na=False
                )
            ]

        if params.get("surface_min") is not None:
            filtered = filtered[
                filtered["Surface reelle bati"] >= params["surface_min"]
            ]

        if params.get("surface_max") is not None:
            filtered = filtered[
                filtered["Surface reelle bati"] <= params["surface_max"]
            ]

        filtered = filtered[filtered["Type local"] == "Maison"]

        if len(filtered) == 0:
            return {
                "success": False,
                "message": "No properties found matching your criteria.",
                "count": 0,
            }

        query_type = params.get("query_type", "mean")
        prices = filtered["last_price"].dropna()

        if len(prices) == 0:
            return {
                "success": False,
                "message": "No price data available for matching properties.",
                "count": len(filtered),
            }

        result = {"success": True, "count": len(filtered)}

        if query_type == "mean":
            result["value"] = prices.mean()
            result["unit"] = "€"
            result["label"] = "Mean price"
        elif query_type == "median":
            result["value"] = prices.median()
            result["unit"] = "€"
            result["label"] = "Median price"
        elif query_type == "count":
            result["value"] = len(filtered)
            result["unit"] = "properties"
            result["label"] = "Number of properties"
        elif query_type == "min":
            result["value"] = prices.min()
            result["unit"] = "€"
            result["label"] = "Minimum price"
        elif query_type == "max":
            result["value"] = prices.max()
            result["unit"] = "€"
            result["label"] = "Maximum price"

        result["price_per_m2"] = filtered["price_per_m2"].dropna().mean()
        result["surface_avg"] = filtered["Surface reelle bati"].dropna().mean()

        return result
