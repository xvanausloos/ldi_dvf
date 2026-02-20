"""Query parser and executor for natural language DVF data queries."""

import ast
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# Load .env from project root (parent of src/)
project_root = Path(__file__).parent.parent.parent
env_path = project_root / ".env"
load_dotenv(dotenv_path=env_path)

logger = logging.getLogger(__name__)


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


def generate_dataset_semantic_layer(df: pd.DataFrame) -> str:
    """Generate a semantic layer description of the dataset for LLM context."""
    if df is None or len(df) == 0:
        return "Dataset is empty."
    
    total_rows = len(df)
    
    # Key columns description
    columns_info = {
        "Code postal": "French postal code (5 digits, e.g., '13820', '75001')",
        "Commune": "City/commune name (e.g., 'Ensues', 'Paris', 'Marseille')",
        "Type local": "Property type - filter for 'Maison' (house) only",
        "Surface reelle bati": "Built surface area in square meters (numeric)",
        "mutations": "Serialized list of transactions with dates and prices",
        "last_price": "Most recent transaction price in euros (derived from mutations)",
        "price_per_m2": "Price per square meter in euros (derived)",
    }
    
    # Statistics
    communes_sample = df["Commune"].value_counts().head(10).to_dict() if "Commune" in df.columns else {}
    postal_codes_sample = df["Code postal"].value_counts().head(10).to_dict() if "Code postal" in df.columns else {}
    
    semantic_layer = f"""DATASET CONTEXT - DVF (Demandes de Valeurs Foncières) French Real Estate Data:

Dataset Overview:
- Total properties: {total_rows:,}
- Time period: 2020-2025
- Property type: Houses only ("Maison")

Key Columns:
"""
    for col, desc in columns_info.items():
        if col in df.columns:
            semantic_layer += f"- {col}: {desc}\n"
    
    if communes_sample:
        semantic_layer += f"\nTop communes (examples): {', '.join(list(communes_sample.keys())[:5])}\n"
    
    if postal_codes_sample:
        semantic_layer += f"Sample postal codes: {', '.join([str(k) for k in list(postal_codes_sample.keys())[:5]])}\n"
    
    semantic_layer += """
Query Processing Rules:
- Always filter by "Type local" = 'Maison' (house)
- Postal codes are 5-digit strings (e.g., "13820", not 13820)
- Commune names are case-insensitive but preserve original capitalization
- Surface is in square meters (m²)
- Prices are in euros (€)
"""
    
    return semantic_layer


class QueryParser:
    """Parse natural language queries about DVF data using LLM with regex fallback."""

    def __init__(self, use_llm: bool = True, model: str = "gpt-4o-mini", df: pd.DataFrame | None = None):
        """Initialize QueryParser.

        Args:
            use_llm: Whether to use LLM for parsing (falls back to regex if unavailable)
            model: OpenAI model to use (default: gpt-4o-mini)
            df: Optional DataFrame to generate semantic layer context
        """
        self.postal_code: str | None = None
        self.commune: str | None = None
        self.surface_min: float | None = None
        self.surface_max: float | None = None
        self.query_type: str = "mean"
        self.use_llm = use_llm
        self.model = model
        self.client = None
        self.semantic_layer = generate_dataset_semantic_layer(df) if df is not None else ""

        if use_llm and OpenAI is not None:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                try:
                    self.client = OpenAI(api_key=api_key)
                    logger.info(f"LLM initialized with model: {model}")
                    if self.semantic_layer:
                        logger.debug("Semantic layer generated for dataset context")
                except Exception as e:
                    logger.error(f"Failed to initialize OpenAI client: {e}")
                    self.use_llm = False
                    self.client = None
            else:
                logger.warning("OPENAI_API_KEY not found in environment. Falling back to regex parsing.")
                self.use_llm = False
        elif use_llm and OpenAI is None:
            logger.warning("OpenAI package not installed. Falling back to regex parsing.")
            self.use_llm = False

    def _parse_with_llm(self, query: str) -> dict[str, Any] | None:
        """Parse query using LLM. Returns None if LLM is unavailable."""
        if not self.client:
            logger.debug("LLM client not available, skipping LLM parsing")
            return None

        logger.info(f"Calling LLM (model: {self.model}) for query: {query[:50]}...")

        # Build system prompt with semantic layer if available
        base_prompt = """You are a helpful assistant that extracts structured information from natural language queries about French real estate data.

Extract the following information from user queries:
- postal_code: French postal code (5 digits) if mentioned, otherwise null
- commune: Commune name (e.g., "Paris", "Marseille", "Ensues") if mentioned, otherwise null
- surface: Surface area in square meters if mentioned (extract the number), otherwise null
- query_type: One of "mean", "median", "count", "min", "max" based on the query intent

Return ONLY valid JSON in this exact format:
{
  "postal_code": "13820" or null,
  "commune": "Ensues" or null,
  "surface": 100.0 or null,
  "query_type": "mean"
}

Examples:
- "What is the mean price of a 100m² house in 13820 Ensues?" -> {"postal_code": "13820", "commune": "Ensues", "surface": 100.0, "query_type": "mean"}
- "How many houses are in Marseille?" -> {"postal_code": null, "commune": "Marseille", "surface": null, "query_type": "count"}
- "What is the median price of houses in Paris?" -> {"postal_code": null, "commune": "Paris", "surface": null, "query_type": "median"}
- "What is the average price of a 80m² house in 75001?" -> {"postal_code": "75001", "commune": null, "surface": 80.0, "query_type": "mean"}

Return ONLY the JSON object, no other text."""

        if self.semantic_layer:
            system_prompt = f"""{self.semantic_layer}

{base_prompt}"""
        else:
            system_prompt = base_prompt

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query},
                ],
                temperature=0.1,
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content
            logger.debug(f"LLM response: {content}")
            parsed = json.loads(content)

            result = {
                "postal_code": parsed.get("postal_code"),
                "commune": parsed.get("commune"),
                "surface_min": None,
                "surface_max": None,
                "query_type": parsed.get("query_type", "mean"),
            }

            if parsed.get("surface") is not None:
                surface = float(parsed["surface"])
                result["surface_min"] = surface * 0.9
                result["surface_max"] = surface * 1.1

            logger.info(f"LLM parsing successful: {result}")
            return result
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM JSON response: {e}")
            return None
        except Exception as e:
            logger.error(f"LLM parsing failed: {e}", exc_info=True)
            return None

    def _parse_with_regex(self, query: str) -> dict[str, Any]:
        """Parse query using regex patterns (fallback method)."""
        postal_code = extract_postal_code(query)
        commune = extract_commune(query)
        surface = extract_surface(query)
        query_type = extract_query_type(query)

        result = {
            "postal_code": postal_code,
            "commune": commune,
            "surface_min": None,
            "surface_max": None,
            "query_type": query_type,
        }

        if surface:
            result["surface_min"] = surface * 0.9
            result["surface_max"] = surface * 1.1

        return result

    def parse(self, query: str) -> tuple[dict[str, Any], bool]:
        """Parse a natural language query and extract parameters.

        Uses LLM if available, otherwise falls back to regex parsing.
        
        Returns:
            tuple: (parsed_params_dict, used_llm_bool)
        """
        if self.use_llm:
            logger.debug(f"Attempting LLM parsing for query: {query}")
            llm_result = self._parse_with_llm(query)
            if llm_result is not None:
                logger.info("Using LLM parsing result")
                self.postal_code = llm_result["postal_code"]
                self.commune = llm_result["commune"]
                self.surface_min = llm_result["surface_min"]
                self.surface_max = llm_result["surface_max"]
                self.query_type = llm_result["query_type"]
                return llm_result, True
            else:
                logger.warning("LLM parsing returned None, falling back to regex")

        logger.debug("Using regex parsing")
        regex_result = self._parse_with_regex(query)
        self.postal_code = regex_result["postal_code"]
        self.commune = regex_result["commune"]
        self.surface_min = regex_result["surface_min"]
        self.surface_max = regex_result["surface_max"]
        self.query_type = regex_result["query_type"]
        return regex_result, False


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

    def generate_sql(self, params: dict[str, Any]) -> str:
        """Generate equivalent SQL query from parameters.
        
        Note: This generates SQL-like syntax. Actual execution uses pandas DataFrame operations.
        Column names with spaces are quoted for SQL compatibility.
        """
        query_type = params.get("query_type", "mean")
        
        # Determine SELECT clause
        if query_type == "count":
            select_clause = "COUNT(*) AS value"
        elif query_type == "mean":
            select_clause = "AVG(last_price) AS value"
        elif query_type == "median":
            # Note: PERCENTILE_CONT is PostgreSQL/ANSI SQL. For other DBs, use APPROX_PERCENTILE or similar
            select_clause = "PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY last_price) AS value"
        elif query_type == "min":
            select_clause = "MIN(last_price) AS value"
        elif query_type == "max":
            select_clause = "MAX(last_price) AS value"
        else:
            select_clause = "AVG(last_price) AS value"

        # Build WHERE conditions (quote column names with spaces)
        conditions = ['"Type local" = \'Maison\'']
        
        if params.get("postal_code"):
            conditions.append(f'"Code postal" = \'{params["postal_code"]}\'')
        
        if params.get("commune"):
            # Escape single quotes in commune name
            commune_escaped = params["commune"].replace("'", "''")
            conditions.append(f'UPPER("Commune") LIKE UPPER(\'%{commune_escaped}%\')')
        
        if params.get("surface_min") is not None:
            conditions.append(f'"Surface reelle bati" >= {params["surface_min"]}')
        
        if params.get("surface_max") is not None:
            conditions.append(f'"Surface reelle bati" <= {params["surface_max"]}')
        
        conditions.append("last_price IS NOT NULL")
        
        where_clause = " AND ".join(conditions)
        
        sql = f"""SELECT {select_clause}
FROM dvf_properties
WHERE {where_clause};"""
        
        return sql
