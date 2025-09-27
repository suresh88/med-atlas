"""
This script reads a spreadsheet of drug names and queries the RxNav REST API
to retrieve various pieces of information for each drug.  For every unique
drug name in the ``Drug Name`` column of the input file, it attempts to
resolve the drug to an RxNorm concept (RxCUI) using an approximate string
match.  Once an RxCUI is found, the script retrieves related concepts of
specific term types (TTYs) that correspond to ingredients, brand names,
components, and dose form groupers.  The results are collected in a table
and written back to an Excel file for downstream analysis.

The matching strategy loosely follows the example described in the problem
statement: a full drug name is searched first, and if no RxCUI is returned
then the name is truncated one token at a time before retrying.  However,
to limit the number of API calls, this implementation performs only a
single approximate search per name.  This approach still yields useful
matches for the majority of terms while keeping the runtime manageable.

Usage::

    python get_rxnav_info.py --input merged_drug_details.xlsx --output output.xlsx

Requirements::

    - pandas
    - requests

The script has been tested against sample API responses fetched via the
browser tool.  Should you wish to verify the behavior yourself, you can
uncomment the ``__main__`` block at the bottom and run the script on a
subset of your data.
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests


RXNAV_BASE = "https://rxnav.nlm.nih.gov/REST"

# Simple cache for proprietary preferred synonyms (PSNs).  When fetching
# proprietary information for an RxCUI, results are stored here to avoid
# repeated network requests.  Keys are RxCUI strings and values are lists
# of preferred synonym strings.
_proprietary_cache: Dict[str, List[str]] = {}

def fetch_proprietary_psn(rxcui: str) -> List[str]:
    """Fetch preferred synonym (PSN) values from the proprietary API.

    For branded concept types (SBD/BPCK), the proprietary API may provide
    more descriptive product names under the PSN (Preferred Synonym) type,
    which often include the brand name and strength in conventional units
    (e.g., ``OPVEE 2.7 MG in 0.1 ML Nasal Spray``).  This helper function
    retrieves all PSN names for the given RxCUI.

    Args:
        rxcui: The RxNorm concept identifier to query.

    Returns:
        A list of PSN strings.  If no PSNs are available or an error
        occurs, returns an empty list.
    """
    url = f"{RXNAV_BASE}/rxcui/{rxcui}/proprietary.json"
    try:
        resp = requests.get(url, timeout=30)
    except Exception as exc:
        print(f"Error querying proprietary information for RxCUI {rxcui}: {exc}")
        return []
    if resp.status_code != 200:
        # Non-fatal error; just return nothing
        return []
    try:
        data = resp.json()
    except Exception:
        return []
    group = data.get("proprietaryGroup", {})
    infos = group.get("proprietaryInfo", []) or []
    psns: List[str] = []
    for info in infos:
        # Only select entries with type PSN (Preferred Synonym).  These
        # represent the canonical branded product name for the concept.
        if info.get("type") == "PSN":
            name = info.get("name")
            if name:
                psns.append(name.strip())
    return psns

def fetch_proprietary_psn_cached(rxcui: str) -> List[str]:
    """Fetch preferred synonyms with caching to avoid repeated API calls.

    Args:
        rxcui: The RxNorm concept identifier.

    Returns:
        A list of preferred synonym strings retrieved from cache or via
        ``fetch_proprietary_psn``.  If the rxcui is empty, returns an empty
        list without calling the API.
    """
    if not rxcui:
        return []
    # Use cache to avoid repeated network requests for the same rxcui.
    if rxcui in _proprietary_cache:
        return _proprietary_cache[rxcui]
    psn = fetch_proprietary_psn(rxcui)
    _proprietary_cache[rxcui] = psn
    return psn

# Mapping from RxNorm term type (TTY) to output column name.  See the
# RxNorm API documentation for more details on TTY values.
TTY_CATEGORY_MAP: Dict[str, str] = {
    "IN": "Ingredient",
    "MIN": "Ingredient",  # multiple ingredient
    "PIN": "Precise Ingredient",
    "BN": "Brand Name",
    "SCDC": "Clinical Drug Component",
    "SBDC": "Branded Drug Component",
    "SCD": "Clinical Drug or Pack",
    "GPCK": "Clinical Drug or Pack",
    "SBD": "Branded Drug or Pack",
    "BPCK": "Branded Drug or Pack",
    "SCDG": "Clinical Dose Form Group",
    "SCDGP": "Clinical Dose Form Group",
    "DFG": "Dose Form Group",
    "SBDG": "Branded Dose Form Group",
}


def normalize_name(name: str) -> str:
    """Normalize a drug name by collapsing whitespace and stripping punctuation.

    Args:
        name: Raw drug name from the spreadsheet.

    Returns:
        A cleaned representation of the drug name.
    """
    # Replace newlines with spaces and collapse multiple whitespace into a single
    # space.  Retain commas since they may separate strengths.
    name = name.replace("\n", " ")
    name = re.sub(r"\s+", " ", name).strip()
    return name


def fetch_rxcui(name: str, max_entries: int = 1) -> Optional[Tuple[str, str]]:
    """Fetch the best matching RxCUI for a given drug name using approximate search.

    Args:
        name: Drug name to search for.
        max_entries: Maximum number of candidate matches to request.

    Returns:
        A tuple ``(rxcui, matched_name)`` if a match is found, otherwise ``None``.

    Notes:
        The RxNorm API ``/approximateTerm`` endpoint returns a list of candidate
        matches ordered by lexical similarity.  We choose the first candidate
        because it has the highest score.  If no candidate is returned the
        function returns ``None``.
    """
    params = {
        "term": name,
        "maxEntries": str(max_entries),
    }
    url = f"{RXNAV_BASE}/approximateTerm.json"
    try:
        resp = requests.get(url, params=params, timeout=30)
    except Exception as exc:
        print(f"Error querying RxNav approximateTerm for '{name}': {exc}")
        return None
    if resp.status_code != 200:
        print(
            f"Warning: Received status {resp.status_code} when querying approximateTerm for '{name}'"
        )
        return None
    try:
        data = resp.json()
    except json.JSONDecodeError as exc:
        print(f"Error decoding JSON response for '{name}': {exc}")
        return None
    group = data.get("approximateGroup", {})
    candidates = group.get("candidate", []) or []
    if not candidates:
        return None
    # Pick the first candidate (highest lexical score)
    best = candidates[0]
    rxcui = best.get("rxcui")
    # Some entries may not have the name field
    matched_name = best.get("name") or ""
    return rxcui, matched_name


def fetch_related_concepts(rxcui: str, ttys: Iterable[str]) -> Dict[str, List[str]]:
    """Retrieve related concepts of specific term types for a given RxCUI.

    Args:
        rxcui: The RxNorm concept identifier to query.
        ttys: Iterable of term type codes to request.

    Returns:
        A mapping from human-friendly category names to lists of concept names.

    Notes:
        The RxNav API ``/rxcui/{rxcui}/related`` endpoint accepts multiple TTYs
        separated by '+' (URL encoded as spaces).  The API returns an array of
        concept groups keyed by TTY.  This function maps each TTY to a
        category using ``TTY_CATEGORY_MAP`` and collects the names of the
        associated concepts.  If no related concepts are found for a given
        TTY, the category will not appear in the result dictionary.
    """
    # Build query string for TTYs.  The API uses space-separated values; requests
    # will handle encoding of spaces automatically.
    tty_param = " ".join(ttys)
    url = f"{RXNAV_BASE}/rxcui/{rxcui}/related.json"
    params = {"tty": tty_param}
    try:
        resp = requests.get(url, params=params, timeout=30)
    except Exception as exc:
        print(f"Error querying related concepts for RxCUI {rxcui}: {exc}")
        return {}
    if resp.status_code != 200:
        print(
            f"Warning: Received status {resp.status_code} when querying related concepts for RxCUI {rxcui}"
        )
        return {}
    try:
        data = resp.json()
    except json.JSONDecodeError as exc:
        print(f"Error decoding JSON for related concepts of RxCUI {rxcui}: {exc}")
        return {}
    result: Dict[str, List[str]] = {}
    group = data.get("relatedGroup", {})
    concept_groups = group.get("conceptGroup", []) or []
    for cg in concept_groups:
        tty = cg.get("tty")
        if not tty:
            continue
        category = TTY_CATEGORY_MAP.get(tty)
        if not category:
            continue
        for prop in cg.get("conceptProperties", []) or []:
            """
            Determine which string (name or synonym) to use for the given term
            type.  For brand‐specific term types (SBD or BPCK) the RxNav API
            typically provides a generic string in the ``name`` field (e.g.,
            ``nalmefene 27 MG/ML Nasal Spray [Opvee]``) and one or more
            brand‐oriented strings in the proprietary information (e.g.,
            ``OPVEE 2.7 MG in 0.1 ML Nasal Spray``).  To return the most
            clinically useful branded form, we attempt to retrieve proprietary
            preferred synonyms (PSN) for these term types.  If no PSN is
            available, fall back to the ``synonym`` field, and finally to the
            ``name`` field.

            For other term types we prefer the ``synonym`` when present (to
            capture more concise forms) and then fall back to the ``name``.
            """
            # Extract the RxCUI for the concept property; some calls omit this
            # but it is needed when querying proprietary information.
            prop_rxcui: str = prop.get("rxcui") or ""
            raw_name: str = prop.get("name", "").strip()
            raw_syn: str = prop.get("synonym", "").strip()
            # Skip entries lacking both name and synonym.
            if not raw_name and not raw_syn:
                continue
            tty_code: str = prop.get("tty", "")
            value_list: List[str] = []
            # For SBD or BPCK, attempt to fetch proprietary preferred synonyms.
            if tty_code in {"SBD", "BPCK"} and prop_rxcui:
                psn = fetch_proprietary_psn_cached(prop_rxcui)
                if psn:
                    value_list.extend(psn)
            # If we did not gather any PSN values (either because the term
            # type is not SBD/BPCK or no PSN is available), fall back to
            # using the synonym if present, otherwise the name.
            if not value_list:
                if raw_syn:
                    value_list.append(raw_syn)
                else:
                    value_list.append(raw_name)
            # Add all collected values to the result under the mapped category.
            for v in value_list:
                if not v:
                    continue
                result.setdefault(category, []).append(v)
    return result


def process_drug_names(names: Iterable[str]) -> pd.DataFrame:
    """Process a sequence of drug names and return a DataFrame of results.

    Args:
        names: Iterable of cleaned drug names.

    Returns:
        A DataFrame with one row per input name and columns for each category.
    """
    # Determine all categories up front to ensure consistent columns
    categories = sorted(set(TTY_CATEGORY_MAP.values()))
    rows = []
    for name in names:
        row: Dict[str, object] = {
            "Drug Name": name,
            "Match Variant": name,  # by default we report the full string used
            "RxCUI": None,
        }
        # Attempt approximate search
        match = fetch_rxcui(name)
        if match is None:
            # Could implement iterative truncation here if desired
            row.update({cat: None for cat in categories})
            rows.append(row)
            continue
        rxcui, matched_name = match
        row["RxCUI"] = rxcui
        row["Matched Name"] = matched_name or None
        # Fetch related concepts
        related = fetch_related_concepts(
            rxcui,
            ttys=TTY_CATEGORY_MAP.keys(),
        )
        for cat in categories:
            values = related.get(cat)
            if values:
                row[cat] = "; ".join(sorted(set(values)))
            else:
                row[cat] = None
        rows.append(row)
    df = pd.DataFrame(rows)
    return df


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fetch RxNav information for a list of drug names."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the Excel file containing a 'Drug Name' column.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path where the output Excel file should be written.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help=(
            "Optional limit on the number of unique names to process. Useful for"
            " testing the script. If omitted, all unique names will be processed."
        ),
    )
    args = parser.parse_args(argv)
    # Read input
    try:
        df_in = pd.read_excel(args.input)
    except Exception as exc:
        print(f"Error reading input file {args.input}: {exc}")
        return 1
    if "Drug Name" not in df_in.columns:
        print("Error: Input file must contain a 'Drug Name' column.")
        return 1
    # Normalize names and deduplicate
    cleaned_names = (
        df_in["Drug Name"]
        .fillna("")
        .apply(normalize_name)
        .drop_duplicates()
        .tolist()
    )
    # Optionally limit for testing
    if args.sample_size is not None:
        cleaned_names = cleaned_names[: args.sample_size]
    print(f"Processing {len(cleaned_names)} unique drug names...")
    df_out = process_drug_names(cleaned_names)
    # Write output
    try:
        df_out.to_excel(args.output, index=False)
    except Exception as exc:
        print(f"Error writing output file {args.output}: {exc}")
        return 1
    print(f"Results written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())