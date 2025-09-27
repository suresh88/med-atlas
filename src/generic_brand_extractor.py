"""Generic and brand name extractor for pharmaceutical datasets.

This utility loads an Excel workbook containing a ``Drug Name`` column and
produces two new columns – ``Generic Name`` and ``Brand Name`` – by parsing
free‑form drug descriptions.  Every record is routed through a GPT model
(``gpt-4o-mini`` by default) to obtain a structured extraction, while a
lightweight heuristic layer cleans and validates the response (for example,
removing dosage-form descriptors such as “intravenous solution”).  Each row
receives a confidence score and annotation of the extraction method, and a
summary report captures overall quality metrics.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

try:  # Optional dependency for GPT fallback
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional
    OpenAI = None

LOGGER = logging.getLogger(__name__)
if not LOGGER.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    LOGGER.addHandler(handler)
LOGGER.setLevel(logging.INFO)

GENERIC_MARKERS_RE = re.compile(
    r"\b(hcl|hydrochloride|monohydrate|hemihydrate|sodium|potassium|"
    r"chloride|acetate|tartrate|sulfate|succinate|fumarate|bitartrate|"
    r"carbonate|mg|mcg|unit|iu|tablet|tab|capsule|caplet|solution|"
    r"suspension|injection|ointment|cream|gel|patch|spray|extended|"
    r"xr|er|sr|dr|ir|oral|topical|ophthalmic|intravenous|iv|im|"
    r"subcutaneous|sc|inhalation|powder|elixir|syrup|drops|lozenge|"
    r"film|lotion|ophth|oph|ophth.soln|kit|pen|prefilled)\b",
    re.IGNORECASE,
)

BRAND_MARKERS_RE = re.compile(r"\b(\d+\s*mg|\d+\s*mcg|\d+\s*iu|tm|®|™)\b", re.IGNORECASE)

BRAND_HINT_WORDS = {
    "brand": 0.5,
    "trade": 0.4,
    "trademark": 0.4,
    "otc": 0.25,
    "rx": 0.1,
    "patent": 0.15,
}

GENERIC_HINT_WORDS = {
    "generic": 0.6,
    "active": 0.25,
    "ingredient": 0.25,
}


@dataclass
class ExtractionResult:
    generic: str = ""
    brand_names: List[str] = field(default_factory=list)
    confidence: float = 0.0
    method: str = "heuristic"
    notes: str = ""


class GenericBrandExtractor:
    """Parse generic and brand components from free‑form drug names."""

    SEPARATOR_PATTERNS = (
        (re.compile(r"\s*/\s*"), "|"),
        (re.compile(r"\s+[-–—]\s+"), "|"),
        (re.compile(r"\s+\+\s+"), "|"),
        (re.compile(r"\s+&\s+"), "|"),
        (re.compile(r"\s+aka\s+", re.IGNORECASE), "|"),
        (re.compile(r"\s+a\.k\.a\.\s+", re.IGNORECASE), "|"),
        (re.compile(r"\s+also known as\s+", re.IGNORECASE), "|"),
        (re.compile(r"\s+w/\s+", re.IGNORECASE), "|"),
        (re.compile(r"\s+with\s+", re.IGNORECASE), "|"),
        (re.compile(r"\s+contains\s+", re.IGNORECASE), "|"),
        (re.compile(r"\s+or\s+", re.IGNORECASE), "|"),
    )

    def __init__(
        self,
        use_gpt: bool = False,
        gpt_model: str = "gpt-4o-mini",
        temperature: float = 0.0,
    ) -> None:
        self.use_gpt = use_gpt and OpenAI is not None and bool(os.getenv("OPENAI_API_KEY"))
        if use_gpt and not self.use_gpt:
            LOGGER.warning("GPT fallback requested but OpenAI dependencies or API key are unavailable. Using heuristics only.")
        self.model = gpt_model
        self.temperature = temperature
        self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) if self.use_gpt else None
        self._gpt_cache: Dict[str, ExtractionResult] = {}

    # ------------------------------------------------------------------
    # Public API
    def parse_dataframe(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        if "Drug Name" not in df.columns:
            raise ValueError("Input DataFrame must contain a 'Drug Name' column.")
        parsed: List[ExtractionResult] = []
        start = time.perf_counter()
        for raw in df["Drug Name"].astype(str).fillna(""):
            parsed.append(self.parse_entry(raw.strip()))
        enriched = df.copy()
        enriched["Generic Name"] = [item.generic for item in parsed]
        enriched["Brand Name"] = ["; ".join(item.brand_names) for item in parsed]
        enriched["Extraction Confidence"] = [round(item.confidence, 3) for item in parsed]
        enriched["Extraction Method"] = [item.method for item in parsed]
        enriched["Extraction Notes"] = [item.notes for item in parsed]
        summary = self.build_summary(parsed, time.perf_counter() - start)
        return enriched, summary

    def parse_entry(self, entry: str) -> ExtractionResult:
        if not entry:
            return ExtractionResult(notes="empty entry")
        heuristic = self._heuristic_parse(entry)
        if heuristic.confidence >= 0.7 or not self.use_gpt:
            return heuristic
        gpt_result = self._gpt_parse(entry)
        if gpt_result:
            return ExtractionResult(
                generic=gpt_result.generic or heuristic.generic,
                brand_names=gpt_result.brand_names or heuristic.brand_names,
                confidence=max(heuristic.confidence, gpt_result.confidence),
                method="heuristic+gpt" if heuristic.generic or heuristic.brand_names else "gpt",
                notes=gpt_result.notes or heuristic.notes,
            )
        return heuristic

    # ------------------------------------------------------------------
    # Heuristic parser
    def _heuristic_parse(self, entry: str) -> ExtractionResult:
        cleaned = self._normalise(entry)
        generics: List[str] = []
        brands: List[str] = []
        notes: List[str] = []

        paren = re.match(r"^(?P<outer>[^()]+?)\s*\((?P<inner>[^()]+)\)\s*$", cleaned)
        if paren:
            outer_profile = self._profile(paren.group("outer"))
            inner_profiles = [self._profile(item) for item in self._split(paren.group("inner"))]
            if outer_profile["generic_score"] >= outer_profile["brand_score"]:
                generics.append(outer_profile["clean"])
                brands.extend(profile["clean"] for profile in inner_profiles)
            else:
                brands.append(outer_profile["clean"])
                if inner_profiles:
                    inner_profiles.sort(
                        key=lambda prof: (
                            round(prof["generic_score"] - prof["brand_score"], 3),
                            -len(prof["clean"]),
                        ),
                        reverse=True,
                    )
                    best = inner_profiles[0]
                    if best["generic_score"] >= best["brand_score"]:
                        generics.append(best["clean"])
                    for profile in inner_profiles[1:]:
                        score_gap = profile["brand_score"] - profile["generic_score"]
                        if score_gap >= -0.1:
                            brands.append(profile["clean"])
            notes.append("parenthetical pattern")
        else:
            segments = self._split(cleaned)
            if len(segments) > 1:
                profiles = [self._profile(segment) for segment in segments]
                profiles.sort(
                    key=lambda prof: (
                        round(prof["generic_score"] - prof["brand_score"], 3),
                        -len(prof["clean"]),
                    ),
                    reverse=True,
                )
                if profiles and profiles[0]["generic_score"] >= profiles[0]["brand_score"]:
                    generics.append(profiles[0]["clean"])
                for profile in profiles[1:]:
                    score_gap = profile["brand_score"] - profile["generic_score"]
                    if score_gap >= -0.05:
                        brands.append(profile["clean"])
                notes.append("multi-part split")
            else:
                profile = self._profile(cleaned)
                if profile["generic_score"] >= profile["brand_score"]:
                    generics.append(profile["clean"])
                    notes.append("single generic candidate")
                else:
                    brands.append(profile["clean"])
                    notes.append("single brand candidate")

        generics = self._dedupe(generics)
        brands = self._dedupe(brands)
        confidence = self._confidence(generics, brands, entry)
        return ExtractionResult(
            generic=generics[0] if generics else "",
            brand_names=brands,
            confidence=confidence,
            method="heuristic",
            notes=", ".join(notes),
        )

    def _normalise(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    def _split(self, text: str) -> List[str]:
        normalised = text
        for pattern, replacement in self.SEPARATOR_PATTERNS:
            normalised = pattern.sub(replacement, normalised)
        parts = [self._normalise(part) for part in re.split(r"[|;,]", normalised)]
        return [part for part in parts if part]

    def _profile(self, candidate: str) -> Dict[str, Any]:
        clean = candidate.strip().strip('-').strip()
        tokens = [tok for tok in re.split(r"[\s\-/]+", clean) if tok]
        total = max(len(tokens), 1)
        lower_ratio = sum(1 for tok in tokens if tok.islower()) / total
        title_ratio = sum(1 for tok in tokens if tok[:1].isupper()) / total
        generic_marker = bool(GENERIC_MARKERS_RE.search(clean.lower()))
        has_digits = bool(re.search(r"\d", clean))
        brand_marker = bool(BRAND_MARKERS_RE.search(clean.lower()))

        generic_score = 0.45 if generic_marker else 0.0
        generic_score += 0.25 * lower_ratio
        generic_score += 0.1 if has_digits else 0.0
        generic_score += 0.1 if len(clean.split()) >= 2 else 0.0
        generic_score += sum(GENERIC_HINT_WORDS.get(tok.lower(), 0.0) for tok in tokens)

        brand_score = 0.4 * title_ratio
        brand_score += 0.2 if not generic_marker else -0.2
        brand_score += 0.15 if len(clean.split()) <= 3 else 0.0
        brand_score -= 0.1 if has_digits else 0.0
        brand_score += 0.15 if brand_marker else 0.0
        brand_score += sum(BRAND_HINT_WORDS.get(tok.lower(), 0.0) for tok in tokens)

        generic_score = max(0.0, min(1.0, generic_score))
        brand_score = max(0.0, min(1.0, brand_score))
        return {"clean": clean, "generic_score": generic_score, "brand_score": brand_score}

    def _confidence(self, generics: Sequence[str], brands: Sequence[str], original: str) -> float:
        if not generics and not brands:
            return 0.25
        if generics and brands:
            spread = min(len(brands), 3)
            base = 0.72 + 0.04 * spread
            if any(len(name.split()) >= 3 for name in generics):
                base += 0.02
            if any(marker in original for marker in ("/", "+", "&")):
                base += 0.01
            return min(base, 0.9)
        if generics or brands:
            return 0.52
        return 0.25

    @staticmethod
    def _dedupe(items: Sequence[str]) -> List[str]:
        seen: set[str] = set()
        result: List[str] = []
        for item in items:
            key = item.lower()
            if key not in seen:
                seen.add(key)
                result.append(item)
        return result

    # ------------------------------------------------------------------
    # GPT fallback
    def _gpt_parse(self, entry: str) -> Optional[ExtractionResult]:
        if not self.use_gpt or not entry:
            return None
        if entry in self._gpt_cache:
            return self._gpt_cache[entry]

        prompt = (
            "Extract the generic and brand drug names from the entry below. "
            "Return JSON with keys 'generic_name' (string) and 'brand_names' (list of strings). "
            "Use empty values when uncertain. Entry:\n" + entry
        )
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": "You are a precise pharmaceutical data assistant."},
                    {"role": "user", "content": prompt},
                ],
            )
            content = response.choices[0].message.content if response.choices else ""
            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                data = {}
            generic = self._normalise(str(data.get("generic_name", ""))) if isinstance(data, dict) else ""
            brands = data.get("brand_names", []) if isinstance(data, dict) else []
            if isinstance(brands, str):
                brands = [brands]
            brands = [self._normalise(b) for b in brands if b]
            result = ExtractionResult(
                generic=generic,
                brand_names=self._dedupe(brands),
                confidence=0.8,
                method="gpt",
                notes="gpt assisted",
            )
            self._gpt_cache[entry] = result
            return result
        except Exception as exc:  # pragma: no cover - defensive safeguard
            LOGGER.warning("GPT parse failed for '%s': %s", entry, exc)
            self._gpt_cache[entry] = ExtractionResult(notes=f"gpt error: {exc}")
            return None

    # ------------------------------------------------------------------
    # Reporting helpers
    @staticmethod
    def build_summary(results: List[ExtractionResult], elapsed: float) -> Dict[str, Any]:
        total = len(results)
        both = sum(1 for item in results if item.generic and item.brand_names)
        generic_only = sum(1 for item in results if item.generic and not item.brand_names)
        brand_only = sum(1 for item in results if not item.generic and item.brand_names)
        unresolved = total - (both + generic_only + brand_only)
        confidences = sorted(item.confidence for item in results)
        avg_conf = round(sum(confidences) / total, 3) if total else 0.0

        def percentile(p: float) -> float:
            if not confidences:
                return 0.0
            if len(confidences) == 1:
                return round(confidences[0], 3)
            k = (len(confidences) - 1) * p
            f = math.floor(k)
            c = math.ceil(k)
            if f == c:
                return round(confidences[int(k)], 3)
            lower = confidences[f]
            upper = confidences[c]
            return round(lower + (upper - lower) * (k - f), 3)

        return {
            "total_rows": total,
            "both_identified": both,
            "generic_only": generic_only,
            "brand_only": brand_only,
            "unresolved": unresolved,
            "average_confidence": avg_conf,
            "method_breakdown": dict(Counter(item.method for item in results)),
            "confidence_percentiles": {
                "p25": percentile(0.25),
                "p50": percentile(0.5),
                "p75": percentile(0.75),
                "p90": percentile(0.9),
            },
            "elapsed_seconds": round(elapsed, 2),
        }


def process_workbook(
    input_path: str,
    sheet_name: str | int | None = 0,
    output_path: Optional[str] = None,
    use_gpt: bool = False,
) -> Dict[str, Any]:
    extractor = GenericBrandExtractor(use_gpt=use_gpt)
    df = pd.read_excel(input_path, sheet_name=sheet_name)
    if isinstance(df, dict):
        df = df[next(iter(df))]
    enriched, summary = extractor.parse_dataframe(df)
    if output_path:
        output_dir = os.path.dirname(os.path.abspath(output_path))
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        enriched.to_excel(output_path, index=False)
    return summary


def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract generic and brand names from drug listings.")
    parser.add_argument("--input", "-i", default="data/output/optum.xlsx", help="Input Excel file path")
    parser.add_argument(
        "--output",
        "-o",
        default="data/output/optum_with_generic_brand.xlsx",
        help="Output Excel file path",
    )
    parser.add_argument("--sheet", default=0, help="Sheet name or index to load")
    parser.add_argument("--use-gpt", action="store_true", help="Enable GPT fallback for ambiguous rows")
    parser.add_argument("--no-save", action="store_true", help="Skip writing the enriched Excel file")
    return parser.parse_args()


def main() -> None:
    args = _parse_cli_args()
    try:
        sheet = int(args.sheet)
    except (TypeError, ValueError):
        sheet = args.sheet
    summary = process_workbook(
        input_path=args.input,
        sheet_name=sheet,
        output_path=None if args.no_save else args.output,
        use_gpt=args.use_gpt,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
