"""
get_disease_relations_structured.py
===================================

This script reads a spreadsheet of drug names, looks up each drug in
RxNorm/RxClass to find MED‑RT disease relations (e.g. *may_treat*,
*may_prevent*, *ci_with*, *may_diagnose*), retrieves the hierarchical
classification path for each disease, and writes the results to a new
Excel file.  The output is designed to make downstream processing
straightforward: each relation type is represented by a column whose
value is a list of dictionaries.  Each dictionary maps a condition
name to a comma‑separated classification path.  Duplicate conditions
within a relation type are removed.  The script also records the
search term (which may be shorter than the original drug name) that
produced the successful RxCUI match.

Usage::

    python get_disease_relations_structured.py \
        --input merged_drug_details.xlsx --output relations_output.xlsx

Arguments:

``--input``
    Path to the input Excel workbook containing a ``Drug Name`` column.

``--output``
    Path to the output Excel workbook where results will be saved.

``--sheet``
    Optional sheet name in the input workbook; defaults to the first sheet.

``--sleep``
    Optional delay in seconds between network calls to avoid server
    throttling (default is 0.1 seconds).

RxClass references:

* ``getClassByRxNormDrugId`` / ``class/byRxcui`` endpoint returns
  drug-class relations, including MED‑RT relations such as
  ``may_treat`` or ``ci_with``【245883907848235†L190-L201】.
* ``classGraph`` endpoint returns the ancestry of a disease class,
  allowing reconstruction of the classification path【917083341744863†L0-L8】.

"""

import argparse
import json
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Set

import pandas as pd
import requests


# API endpoints
RXNAV_APPROX_URL = "https://rxnav.nlm.nih.gov/REST/approximateTerm.json"
RXCLASS_BY_RXCUI_URL = "https://rxnav.nlm.nih.gov/REST/rxclass/class/byRxcui.json"
RXCLASS_GRAPH_URL = "https://rxnav.nlm.nih.gov/REST/rxclass/classGraph.json"

# Relations to capture from MED‑RT
RELATIONS_OF_INTEREST = {"may_treat", "may_prevent", "ci_with", "may_diagnose"}

# Common root segments to remove from hierarchy
COMMON_ROOT = [
    "Disease",
    "Diseases, Life Phases, Behavior Mechanisms and Physiologic States",
]


def normalize_name(name: str) -> str:
    """Normalize a drug name by stripping and condensing whitespace and lowercasing."""
    return " ".join(name.strip().split()).lower()


def find_rxcui_with_variants(drug_name: str, max_candidates: int = 20, sleep: float = 0.1) -> tuple[Optional[str], Optional[str]]:
    """
    Attempt to find the RxCUI for a drug by progressively shortening the
    name.  Returns a tuple of ``(rxcui, search_term)``; ``search_term`` is
    the term that produced the match.  If no match is found, both
    elements are ``None``.
    """
    terms = drug_name.split()
    while terms:
        term = " ".join(terms)
        params = {"term": term, "maxEntries": max_candidates}
        try:
            resp = requests.get(RXNAV_APPROX_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            return None, None
        candidates = data.get("approximateGroup", {}).get("candidate", [])
        if candidates:
            return candidates[0].get("rxcui"), term
        # drop the last word and retry
        terms = terms[:-1]
        time.sleep(sleep)
    return None, None


def get_medrt_relations(rxcui: str, sleep: float = 0.1) -> List[tuple[str, str, str]]:
    """
    For a given RxCUI, fetch MED‑RT disease relations and return a list of
    (classId, conditionName, relation) tuples.  Only relations in
    ``RELATIONS_OF_INTEREST`` are returned.
    """
    try:
        resp = requests.get(RXCLASS_BY_RXCUI_URL, params={"rxcui": rxcui}, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return []
    relations: List[tuple[str, str, str]] = []
    entries = (
        data.get("rxclassDrugInfoList", {}).get("rxclassDrugInfo", [])
        if isinstance(data, dict)
        else []
    )
    for entry in entries:
        source = entry.get("relaSource", "").upper()
        relation = entry.get("rela", "").lower()
        if source == "MEDRT" and relation in RELATIONS_OF_INTEREST:
            item = entry.get("rxclassMinConceptItem", {})
            class_id = item.get("classId")
            class_name = item.get("className")
            if class_id and class_name:
                relations.append((class_id, class_name, relation))
    time.sleep(sleep)
    return relations


def get_class_path(class_id: str, cache: Dict[str, List[str]], sleep: float = 0.1) -> List[str]:
    """
    Return the full classification path for a disease class (root→leaf).
    Uses a cache to avoid redundant API calls.
    """
    if class_id in cache:
        return cache[class_id]
    try:
        resp = requests.get(
            RXCLASS_GRAPH_URL,
            params={"classId": class_id, "source": "disease"},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        cache[class_id] = [class_id]
        return cache[class_id]
    graph = data.get("rxclassGraph", {})
    nodes = {n["classId"]: n["className"] for n in graph.get("rxclassMinConceptItem", [])}
    edges = graph.get("rxclassEdge", [])
    parent_map = {edge["classId1"]: edge["classId2"] for edge in edges}
    # Build path from class_id upwards
    path_ids = [class_id]
    current = class_id
    for _ in range(50):
        if current in parent_map:
            current = parent_map[current]
            path_ids.append(current)
            if current == path_ids[-2]:  # self loop
                break
        else:
            break
    path_names = [nodes.get(cid, cid) for cid in reversed(path_ids)]
    cache[class_id] = path_names
    time.sleep(sleep)
    return path_names


def trim_root(path: List[str]) -> str:
    """Remove the common root segments from the beginning of a path and join with commas."""
    trimmed = path.copy()
    while trimmed and trimmed[0] in COMMON_ROOT:
        trimmed.pop(0)
    return ", ".join(trimmed)


def process_drugs(df: pd.DataFrame, drug_col: str, sleep: float) -> pd.DataFrame:
    """
    Process each drug in the dataframe and build a structured representation.
    Returns a new DataFrame suitable for writing to Excel.
    """
    hierarchy_cache: Dict[str, List[str]] = {}
    records: List[Dict[str, object]] = []
    for _, row in df.iterrows():
        raw_name = str(row[drug_col]) if pd.notna(row[drug_col]) else ""
        name_norm = normalize_name(raw_name)
        if not name_norm:
            continue
        rxcui, search_term = find_rxcui_with_variants(name_norm, sleep=sleep)
        if not rxcui:
            continue
        relations = get_medrt_relations(rxcui, sleep=sleep)
        # Initialize dictionary for this drug
        row_dict: Dict[str, object] = {
            "Drug Name": raw_name,
            "Search Term": search_term,
            "RxCUI": rxcui,
        }
        # Prepare containers for each relation type
        relation_values: Dict[str, List[Dict[str, str]]] = {rel: [] for rel in RELATIONS_OF_INTEREST}
        seen_conditions: Dict[str, Set[str]] = {rel: set() for rel in RELATIONS_OF_INTEREST}
        for class_id, condition, relation in relations:
            if condition not in seen_conditions[relation]:
                full_path = get_class_path(class_id, hierarchy_cache, sleep=sleep)
                trimmed = trim_root(full_path)
                relation_values[relation].append({condition: trimmed})
                seen_conditions[relation].add(condition)
        # Fill relation columns (empty lists if no values)
        for rel in RELATIONS_OF_INTEREST:
            row_dict[rel] = relation_values[rel]
        records.append(row_dict)
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract MED‑RT relations and disease hierarchies for drug names."
    )
    parser.add_argument("--input", required=True, help="Input Excel file path")
    parser.add_argument("--output", required=True, help="Output Excel file path")
    parser.add_argument("--sheet", default=None, help="Sheet name in input file")
    parser.add_argument(
        "--sleep", type=float, default=0.1, help="Delay between API calls in seconds"
    )
    args = parser.parse_args()
    # Read the input workbook.  If a sheet name is provided, load that sheet;
    # otherwise rely on pandas' default (sheet_name=0) to return the first sheet.
    try:
        if args.sheet is not None:
            df_in = pd.read_excel(args.input, sheet_name=args.sheet)
        else:
            # omit sheet_name argument to use pandas default (first sheet)
            df_in = pd.read_excel(args.input)
    except Exception as e:
        print(f"Failed to read input file: {e}", file=sys.stderr)
        sys.exit(1)
    # If reading with sheet_name=None returned a dict of DataFrames, default to the first
    if isinstance(df_in, dict):
        # pick the first sheet
        first_key = next(iter(df_in))
        df_in = df_in[first_key]
    if "Drug Name" not in df_in.columns:
        print("Input file must contain a 'Drug Name' column", file=sys.stderr)
        sys.exit(1)
    df_out = process_drugs(df_in, drug_col="Drug Name", sleep=args.sleep)
    try:
        df_out.to_excel(args.output, index=False)
        print(f"Wrote {len(df_out)} rows to {args.output}")
    except Exception as e:
        print(f"Failed to write output file: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
