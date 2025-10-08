"""ICD-10 code extraction powered by GPT models."""

from __future__ import annotations

import argparse
import importlib
import json
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence


DEFAULT_SYSTEM_PROMPT = (
    "You are a certified medical coding specialist. Given disease or condition names, "
    "identify all clinically relevant ICD-10-CM diagnosis codes. Always include the "
    "unspecified code when it exists, and supplement with high-yield specific codes. "
    "If a disease cannot be mapped, return an empty list for that entry. Respond "
    "exclusively in JSON without additional commentary."
)


@dataclass(frozen=True)
class DiseaseMapping:
    """A single disease-to-code mapping returned by the GPT model."""

    disease: str
    icd10_codes: Sequence[str]


class ICD10Mapper:
    """Map diseases to ICD-10 codes using an OpenAI GPT model."""

    def __init__(
        self,
        *,
        model: str = "gpt-4o-mini",
        client: Optional[Any] = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_batch_size: int = 20,
        temperature: float = 0.0,
    ) -> None:
        self.model = model
        self._client = client
        self.system_prompt = system_prompt
        self.max_batch_size = max(1, max_batch_size)
        self.temperature = temperature

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def lookup(self, disease: str) -> List[str]:
        """Return ICD-10 codes for ``disease`` using the configured GPT model."""

        if not disease or not isinstance(disease, str):
            return []
        mappings = self.map_many([disease])
        if not mappings:
            return []
        return list(mappings[0].icd10_codes)

    def map_many(self, diseases: Iterable[str]) -> List[DiseaseMapping]:
        """Return GPT-generated mappings for an iterable of disease names."""

        ordered: List[str] = []
        filtered: List[str] = []
        for disease in diseases:
            if isinstance(disease, str):
                ordered.append(disease)
                if disease.strip():
                    filtered.append(disease)
            elif disease is None:
                ordered.append("")
            else:
                text = str(disease)
                ordered.append(text)
                if text.strip():
                    filtered.append(text)

        if not filtered:
            return [DiseaseMapping(disease=d, icd10_codes=()) for d in ordered]

        unique_order: List[str] = list(dict.fromkeys(filtered))
        mapping: Dict[str, List[str]] = {}
        for start in range(0, len(unique_order), self.max_batch_size):
            batch = unique_order[start : start + self.max_batch_size]
            batch_mapping = self._fetch_batch_mapping(batch)
            mapping.update(batch_mapping)

        results: List[DiseaseMapping] = []
        for disease in ordered:
            if isinstance(disease, str) and disease.strip():
                codes = tuple(mapping.get(disease, []))
            else:
                codes = ()
            results.append(DiseaseMapping(disease=disease, icd10_codes=codes))
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _fetch_batch_mapping(self, diseases: Sequence[str]) -> Dict[str, List[str]]:
        client = self._get_client()
        response = client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "system",
                    "content": [{"type": "text", "text": self.system_prompt}],
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": self._format_user_prompt(diseases)}],
                },
            ],
            response_format=self._build_response_format(),
            temperature=self.temperature,
        )
        return self._parse_response(diseases, response)

    def _format_user_prompt(self, diseases: Sequence[str]) -> str:
        payload = json.dumps({"diseases": list(diseases)}, ensure_ascii=False)
        instructions = (
            "You will receive a JSON object with a \"diseases\" array. "
            "For each disease name, provide the most relevant ICD-10-CM diagnosis codes. "
            "Return a JSON array where every element contains the original disease string "
            "and a list of ICD-10 codes. If a mapping cannot be determined, use an empty list."
        )
        example = (
            "Example response: "
            "[{\"disease\": \"Hypertension\", \"icd10_codes\": [\"I10\"]}]"
        )
        return f"{instructions}\n{example}\n\nInput:\n{payload}"

    @staticmethod
    def _build_response_format() -> Dict[str, Any]:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "icd10_mapping",
                "schema": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["disease", "icd10_codes"],
                        "additionalProperties": False,
                        "properties": {
                            "disease": {"type": "string"},
                            "icd10_codes": {
                                "type": "array",
                                "items": {"type": "string"},
                                "default": [],
                            },
                        },
                    },
                },
            },
        }

    def _get_client(self) -> Any:
        if self._client is None:
            openai_module = importlib.import_module("openai")
            openai_client_factory = getattr(openai_module, "OpenAI")
            self._client = openai_client_factory()
        return self._client

    def _parse_response(
        self, diseases: Sequence[str], response: Any
    ) -> Dict[str, List[str]]:
        text_payload = self._extract_response_text(response)
        try:
            data = json.loads(text_payload)
        except (TypeError, json.JSONDecodeError):
            return {disease: [] for disease in diseases}

        parsed: Dict[str, List[str]] = {disease: [] for disease in diseases}
        for entry in data if isinstance(data, list) else []:
            if not isinstance(entry, dict):
                continue
            disease = entry.get("disease")
            codes = entry.get("icd10_codes", [])
            if not isinstance(disease, str) or disease not in parsed:
                continue
            parsed[disease] = self._normalise_codes(codes)
        return parsed

    @staticmethod
    def _extract_response_text(response: Any) -> str:
        if hasattr(response, "output_text") and isinstance(response.output_text, str):
            return response.output_text
        output = getattr(response, "output", None)
        text_parts: List[str] = []
        if output:
            for item in output:
                content = getattr(item, "content", None) or []
                for segment in content:
                    text = getattr(segment, "text", None)
                    if isinstance(text, str):
                        text_parts.append(text)
                    elif isinstance(segment, dict) and isinstance(segment.get("text"), str):
                        text_parts.append(segment["text"])
        if not text_parts and hasattr(response, "choices"):
            for choice in getattr(response, "choices", []):
                message = getattr(choice, "message", None)
                content = getattr(message, "content", None)
                if isinstance(content, str):
                    text_parts.append(content)
        return "".join(text_parts)

    @staticmethod
    def _normalise_codes(codes: Any) -> List[str]:
        if not isinstance(codes, (list, tuple)):
            return []
        normalised: List[str] = []
        for code in codes:
            if code is None:
                continue
            text = str(code).strip().upper()
            if text:
                normalised.append(text)
        return normalised


def load_may_treat_diseases(excel_path: Path, column: str = "may_treat_diseases") -> List[str]:
    """Extract the flattened list of diseases from the workbook column."""

    import ast
    import pandas as pd

    df = pd.read_excel(excel_path)
    diseases: List[str] = []
    for value in df[column].dropna():
        if isinstance(value, str):
            try:
                parsed = ast.literal_eval(value)
            except (SyntaxError, ValueError):
                continue
            if isinstance(parsed, (list, tuple)):
                diseases.extend(str(item).strip() for item in parsed if str(item).strip())
    return diseases


def build_mapping_for_workbook(
    excel_path: Path, *, mapper: Optional[ICD10Mapper] = None
) -> Dict[str, List[str]]:
    """Generate ICD-10 mappings for every disease present in the workbook."""

    mapper = mapper or ICD10Mapper()
    disease_list = sorted(set(load_may_treat_diseases(excel_path)))
    result: Dict[str, List[str]] = {}
    for mapping in mapper.map_many(disease_list):
        result[mapping.disease] = list(mapping.icd10_codes)
    return result


def map_disease_list(
    diseases: Iterable[str], *, mapper: Optional[ICD10Mapper] = None
) -> List[Dict[str, List[str]]]:
    """Convenience wrapper returning dictionaries for downstream consumers."""

    mapper = mapper or ICD10Mapper()
    mappings = mapper.map_many(diseases)
    return [
        {"disease": mapping.disease, "icd10_codes": list(mapping.icd10_codes)}
        for mapping in mappings
    ]


def extract_diseases_for_drug(
    excel_path: Path,
    drug_query: str,
    *,
    name_column: str = "Drug Name",
    disease_column: str = "may_treat_diseases",
) -> List[str]:
    """Return unique diseases associated with ``drug_query`` in the workbook."""

    if not drug_query:
        return []

    import ast
    import pandas as pd

    try:
        df = pd.read_excel(excel_path, usecols=[name_column, disease_column])
    except ValueError:
        df = pd.read_excel(excel_path)

    if name_column not in df.columns or disease_column not in df.columns:
        return []

    mask = df[name_column].astype(str).str.contains(drug_query, case=False, na=False)
    if not mask.any():
        return []

    diseases: List[str] = []
    for value in df.loc[mask, disease_column].dropna():
        if isinstance(value, str):
            try:
                parsed = ast.literal_eval(value)
            except (SyntaxError, ValueError):
                continue
            if isinstance(parsed, (list, tuple)):
                diseases.extend(str(item).strip() for item in parsed if str(item).strip())

    seen: Dict[str, None] = {}
    for disease in diseases:
        key = disease
        if key not in seen:
            seen[key] = None
    return list(seen.keys())


def _default_workbook_path() -> Path:
    root = Path(__file__).resolve().parents[1]
    return root / "formulary_data" / "rxnav_relations_with_extracted_diseases.xlsx"


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command line interface for looking up ICD-10 codes via GPT."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--excel-path",
        type=Path,
        default=_default_workbook_path(),
        help="Path to the RxNav workbook containing may_treat disease data.",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI model name to use for ICD-10 extraction.",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=20,
        help="Maximum number of diseases to send in a single GPT request.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature used for the GPT request.",
    )

    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument(
        "--drug-name",
        help="Drug name to search for in the workbook. All may_treat diseases will be mapped.",
    )
    target_group.add_argument(
        "--disease",
        action="append",
        dest="diseases",
        help="Explicit disease name to map. Can be passed multiple times.",
    )

    args = parser.parse_args(argv)

    mapper = ICD10Mapper(
        model=args.model,
        max_batch_size=args.max_batch_size,
        temperature=args.temperature,
    )

    if args.drug_name:
        diseases = extract_diseases_for_drug(args.excel_path, args.drug_name)
        if not diseases:
            parser.error(
                f"No diseases found for drug '{args.drug_name}' in {args.excel_path}"
            )
    else:
        diseases = [d for d in args.diseases or [] if d and d.strip()]
        if not diseases:
            parser.error("At least one --disease value is required when --drug-name is omitted.")

    mappings = map_disease_list(diseases, mapper=mapper)
    print(json.dumps(mappings, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "ICD10Mapper",
    "DiseaseMapping",
    "load_may_treat_diseases",
    "build_mapping_for_workbook",
    "map_disease_list",
    "extract_diseases_for_drug",
    "main",
]
