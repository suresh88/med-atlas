import json
import pathlib
import sys
from typing import Any, Dict, List

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from icd10_mapper import (
    ICD10Mapper,
    extract_diseases_for_drug,
    map_disease_list,
)


class _StubResponse:
    def __init__(self, text: str) -> None:
        self.output_text = text


class _StubResponsesAPI:
    def __init__(self, payloads: List[str]) -> None:
        self._payloads = payloads
        self.calls: List[Dict[str, Any]] = []

    def create(self, **kwargs: Any) -> _StubResponse:
        if not self._payloads:
            raise AssertionError("No stubbed responses remaining")
        self.calls.append(kwargs)
        return _StubResponse(self._payloads.pop(0))


class _StubOpenAIClient:
    def __init__(self, payloads: List[str]) -> None:
        self.responses = _StubResponsesAPI(payloads)


def test_lookup_uses_stubbed_openai_client() -> None:
    payload = json.dumps(
        [
            {"disease": "Hypertension", "icd10_codes": ["I10", "I11.9"]},
        ]
    )
    client = _StubOpenAIClient([payload])
    mapper = ICD10Mapper(client=client, model="test-model")

    codes = mapper.lookup("Hypertension")

    assert codes == ["I10", "I11.9"]
    assert client.responses.calls[0]["model"] == "test-model"


def test_map_many_handles_duplicate_diseases() -> None:
    payload = json.dumps(
        [
            {"disease": "Hypertension", "icd10_codes": ["I10"]},
            {"disease": "Diabetes", "icd10_codes": ["E11.9", "E11.65"]},
        ]
    )
    client = _StubOpenAIClient([payload])
    mapper = ICD10Mapper(client=client)

    results = mapper.map_many(["Hypertension", "Diabetes", "Hypertension"])

    assert [tuple(mapping.icd10_codes) for mapping in results] == [
        ("I10",),
        ("E11.9", "E11.65"),
        ("I10",),
    ]
    user_prompt = client.responses.calls[0]["input"][1]["content"][0]["text"]
    assert "Hypertension" in user_prompt and "Diabetes" in user_prompt


def test_map_disease_list_handles_empty_iterable() -> None:
    assert map_disease_list([]) == []


def test_map_disease_list_uses_provided_mapper() -> None:
    payload = json.dumps(
        [
            {"disease": "Condition That Exists", "icd10_codes": ["A00.0"]},
        ]
    )
    client = _StubOpenAIClient([payload])
    mapper = ICD10Mapper(client=client)

    results = map_disease_list(["Condition That Exists"], mapper=mapper)

    assert results == [{"disease": "Condition That Exists", "icd10_codes": ["A00.0"]}]


def test_lookup_returns_empty_when_model_response_missing_disease() -> None:
    payload = json.dumps([])
    client = _StubOpenAIClient([payload])
    mapper = ICD10Mapper(client=client)

    assert mapper.lookup("Unknown Condition") == []


def test_extract_diseases_for_drug_returns_unique_matches(tmp_path: pathlib.Path) -> None:
    import pandas as pd

    data = pd.DataFrame(
        {
            "Drug Name": ["SampleDrug 500mg Tablet", "OtherDrug"],
            "may_treat_diseases": ["['Disease A', 'Disease B', 'Disease A']", "[]"],
        }
    )
    workbook = tmp_path / "workbook.xlsx"
    data.to_excel(workbook, index=False)

    diseases = extract_diseases_for_drug(workbook, "sampledrug")

    assert diseases == ["Disease A", "Disease B"]
