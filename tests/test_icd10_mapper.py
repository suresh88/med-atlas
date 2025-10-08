import pathlib
import sys

import pytest

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from icd10_mapper import ICD10Mapper, map_disease_list


def test_lookup_known_manual_mappings():
    mapper = ICD10Mapper()
    assert set(mapper.lookup("Hypertension")) >= {"I10", "I11.9", "I12.9"}
    assert set(mapper.lookup("Diabetes Mellitus, Type 2")) >= {"E11.9", "E11.65", "E11.69"}


def test_lookup_keyword_matches():
    mapper = ICD10Mapper()
    acidosis_codes = mapper.lookup("Acidosis, Renal Tubular")
    conjunctivitis_codes = mapper.lookup("Conjunctivitis, Bacterial")
    assert "E87.2" in acidosis_codes
    assert any(code.startswith("H10.") for code in conjunctivitis_codes)


def test_map_disease_list_handles_empty_iterable():
    assert map_disease_list([]) == []


def test_lookup_returns_list_even_when_no_match():
    mapper = ICD10Mapper()
    assert mapper.lookup("Condition That Does Not Exist") == []
