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


def get_all_class_paths(class_id: str, cache: Dict[str, List[List[str]]], sleep: float = 0.1) -> List[List[str]]:
    """
    Return all classification paths for a disease class (root→leaf).

    A disease class can belong to multiple branches of the hierarchy.  This
    function retrieves the full directed acyclic graph of relationships
    via the ``classGraph`` endpoint and enumerates every possible path
    from the root down to the class.  The root nodes (``Disease`` and
    ``Diseases, Life Phases, Behavior Mechanisms and Physiologic States``)
    are included in the returned lists and may be removed later.

    Results are cached to avoid redundant network calls.

    Args:
        class_id: The RxClass identifier of the disease.
        cache: A mapping used to memoize previously computed paths.
        sleep: Delay (in seconds) after each API call.

    Returns:
        A list of paths, where each path is a list of class names from
        root to the specified class (inclusive).  If the API call fails,
        returns a single path containing only the class name.
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
        # Fallback: no hierarchy available
        cache[class_id] = [[class_id]]
        return cache[class_id]
    graph = data.get("rxclassGraph", {})
    # Build mapping of classId to className
    nodes: Dict[str, str] = {n["classId"]: n["className"] for n in graph.get("rxclassMinConceptItem", [])}
    # Build parent adjacency list (child -> list of parents)
    parent_map: Dict[str, List[str]] = {}
    for edge in graph.get("rxclassEdge", []):
        child = edge.get("classId1")
        parent = edge.get("classId2")
        if child and parent:
            parent_map.setdefault(child, []).append(parent)

    # Depth-first search to enumerate all paths from class_id up to root
    all_paths: List[List[str]] = []
    def dfs(node: str, path: List[str]):
        path.append(node)
        parents = parent_map.get(node)
        if not parents:
            # Reached a root; record the path (convert ids to names)
            path_names = [nodes.get(cid, cid) for cid in reversed(path)]
            all_paths.append(path_names)
        else:
            for parent in parents:
                if parent not in path:  # avoid cycles
                    dfs(parent, path)
        path.pop()

    dfs(class_id, [])
    # If no paths were found (no edges), return the class itself
    if not all_paths:
        all_paths = [[nodes.get(class_id, class_id)]]
    cache[class_id] = all_paths
    time.sleep(sleep)
    return all_paths


def trim_root_and_leaf(path: List[str]) -> List[str]:
    """Remove common root segments and the final leaf (the condition itself).

    The classification path returned from ``get_all_class_paths`` includes the
    disease class as the last element and the common root(s) as the first
    elements.  This function removes those elements, leaving only the
    intermediate hierarchy.  If the resulting list is empty, it returns
    an empty list.

    Args:
        path: A list of class names from root to the disease (inclusive).

    Returns:
        A list of class names with the root segments and leaf removed.
    """
    trimmed = list(path)
    # Remove root segments
    while trimmed and trimmed[0] in COMMON_ROOT:
        trimmed.pop(0)
    # Remove the final element (leaf/disease) if present
    if trimmed:
        trimmed.pop()  # remove disease itself
    return trimmed


def process_dataframe(df_in: pd.DataFrame, drug_col: str, sleep: float) -> pd.DataFrame:
    """
    Process the input DataFrame and build a structured representation for MED‑RT relations.

    This function preserves all original columns from ``df_in`` and appends
    additional columns: ``Search Term``, ``RxCUI``, and one column per
    relation type defined in ``RELATIONS_OF_INTEREST``.  Each relation
    column contains a list of dictionaries, where the key is the condition
    name and the value is a list of classification paths (each path a
    list of class names).  Duplicate conditions and duplicate paths are
    removed.

    Args:
        df_in: The input DataFrame containing at least the column
            specified by ``drug_col``.
        drug_col: Name of the column with the raw drug names.
        sleep: Delay between API calls in seconds.

    Returns:
        A new DataFrame with the same number of rows as ``df_in`` and
        additional columns containing the extracted relations.
    """
    if drug_col not in df_in.columns:
        raise ValueError(f"Input DataFrame must contain a '{drug_col}' column")
    # Normalize names for grouping
    df = df_in.copy()
    df["_norm_name"] = df[drug_col].fillna("").astype(str).apply(normalize_name)
    # Determine unique normalized names
    unique_norms = df["_norm_name"].unique().tolist()
    # Prepare a cache for classification paths and for per-name results
    hierarchy_cache: Dict[str, List[List[str]]] = {}
    name_result_cache: Dict[str, Dict[str, object]] = {}
    for norm_name in unique_norms:
        if not norm_name:
            # Empty name: no search
            name_result_cache[norm_name] = {
                "Search Term": None,
                "RxCUI": None,
                **{rel: [] for rel in RELATIONS_OF_INTEREST},
            }
            continue
        # Attempt to find RxCUI by truncating name
        rxcui, search_term = find_rxcui_with_variants(norm_name, sleep=sleep)
        # Initialize relation data
        relation_data: Dict[str, List[Dict[str, List[List[str]]]]] = {
            rel: [] for rel in RELATIONS_OF_INTEREST
        }
        if rxcui:
            rels = get_medrt_relations(rxcui, sleep=sleep)
            # Track conditions seen per relation
            seen_cond: Dict[str, Set[str]] = {rel: set() for rel in RELATIONS_OF_INTEREST}
            # For each relation entry, compute all paths
            for class_id, condition, relation in rels:
                if condition in seen_cond[relation]:
                    continue
                all_paths = get_all_class_paths(class_id, hierarchy_cache, sleep=sleep)
                # Trim root and leaf for each path and filter out empty
                trimmed_paths: List[List[str]] = []
                for p in all_paths:
                    tp = trim_root_and_leaf(p)
                    if tp:  # only include non-empty paths
                        trimmed_paths.append(tp)
                # Remove duplicate paths
                unique_tp = []
                seen_tp = set()
                for tp in trimmed_paths:
                    tup = tuple(tp)
                    if tup not in seen_tp:
                        unique_tp.append(tp)
                        seen_tp.add(tup)
                relation_data[relation].append({condition: unique_tp})
                seen_cond[relation].add(condition)
        else:
            search_term = norm_name
        name_result_cache[norm_name] = {
            "Search Term": search_term,
            "RxCUI": rxcui,
            **relation_data,
        }
    # Build output rows by combining original row with cached results
    output_rows: List[Dict[str, object]] = []
    for _, row in df.iterrows():
        norm = row["_norm_name"]
        info = name_result_cache.get(norm, {})
        # Copy original row data
        out_row = row.drop(labels=["_norm_name"]).to_dict()
        # Append search term, rxcui and relation columns
        out_row.update(info)
        # Ensure that all relation columns exist even if missing in info
        for rel in RELATIONS_OF_INTEREST:
            if rel not in out_row:
                out_row[rel] = []
        output_rows.append(out_row)
    # Construct DataFrame from rows
    return pd.DataFrame(output_rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract MED‑RT disease relations for drug names while retaining all input columns."
        )
    )
    parser.add_argument("--input", required=True, help="Input Excel file path")
    parser.add_argument("--output", required=True, help="Output Excel file path")
    parser.add_argument("--sheet", default=None, help="Sheet name in input file")
    parser.add_argument(
        "--sleep", type=float, default=0.1, help="Delay between API calls in seconds"
    )
    args = parser.parse_args()
    # Read the input workbook.  If sheet is specified, load it; otherwise load first sheet.
    try:
        if args.sheet is not None:
            df_in = pd.read_excel(args.input, sheet_name=args.sheet)
        else:
            # Let pandas default to the first sheet
            df_in = pd.read_excel(args.input)
    except Exception as e:
        print(f"Failed to read input file: {e}", file=sys.stderr)
        sys.exit(1)
    # If pandas returned a dict (multi-sheet without specifying sheet_name), pick the first sheet
    if isinstance(df_in, dict):
        df_in = df_in[next(iter(df_in))]
    # Validate that the drug column exists
    if "Drug Name" not in df_in.columns:
        print("Input file must contain a 'Drug Name' column", file=sys.stderr)
        sys.exit(1)
    print(f"Processing {len(df_in)} rows...")
    try:
        df_out = process_dataframe(df_in, drug_col="Drug Name", sleep=args.sleep)
    except Exception as e:
        print(f"Failed to process data: {e}", file=sys.stderr)
        sys.exit(1)
    try:
        df_out.to_excel(args.output, index=False)
        print(f"Wrote {len(df_out)} rows to {args.output}")
    except Exception as e:
        print(f"Failed to write output file: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
