"""
icd_extractor.py
=================

This module contains a single class, ``ICDExtractor``, that reads an Excel
spreadsheet with lists of disease names and maps those diseases to ICD‑10
codes by querying the U.S. National Library of Medicine’s Clinical Table
Search Service (CTSS) API.  The CTSS API indexes the ICD‑10‑CM code set
and allows free‑text lookup of codes and descriptions.

The extractor implements several features:

  * **Excel Processing**: The input Excel file is expected to have the
    columns ``may_prevent_diseases``, ``ci_with_diseases``,
    ``may_diagnose_diseases`` and ``may_treat_diseases``.  Each of these
    columns contains a string representation of a Python list of disease
    names.  The extractor uses ``ast.literal_eval`` to safely parse these
    strings into lists and then collects a unique set of disease names.

  * **API Integration**: The extractor uses the Clinical Table Search
    Service to search for ICD‑10 codes.  According to the official
    documentation, every call must include a ``terms`` parameter with
    the query text and can specify other optional parameters such as
    ``sf`` (search fields) and ``maxList`` (maximum number of results)
   【125092994786881†L40-L60】.  To retrieve codes by disease name we set
    ``sf=code,name`` so the API searches both code and description fields,
    and we increase ``maxList`` from the default of 7 to 20 to retrieve
    more matches.  The API returns a JSON array where the second element
    (index 1) is a list of ICD‑10 codes and the fourth element (index 3)
    is a list of corresponding display strings【125092994786881†L101-L118】.

  * **Caching**:  Because many diseases appear multiple times in the input
    file, the extractor caches results locally.  The cache is stored on
    disk as a JSON file and loaded at initialization.  Each disease
    string maps to a list of codes (possibly empty).  If a disease is
    already in the cache the extractor does not issue another API call.

  * **Fallback Matching**:  Not all disease strings exactly match
    descriptions in the ICD‑10 database.  When a direct lookup returns
    no codes, the extractor falls back to searching substrings of the
    disease.  For example, ``"Arthritis, Rheumatoid"`` might not match
    directly but splitting on commas yields ``"Arthritis"`` and
    ``"Rheumatoid"``, each of which may return codes.  As a last
    resort, the disease is split on whitespace and each token is
    queried.  All unique codes returned by these fallback queries are
    recorded.

  * **Error Handling**:  The extractor is resilient to malformed
    entries (e.g., cells that do not contain a valid list) and to
    transient network errors.  When an API call fails, it retries a few
    times with exponential backoff.  Diseases for which no codes are
    found are still recorded in the cache with an empty list so that
    subsequent runs skip them.

Example
-------
To run the extractor on an Excel file and save the results to a CSV:

.. code-block:: python

    from icd_extractor import ICDExtractor

    extractor = ICDExtractor(cache_file='icd_cache.json')
    df = extractor.load_excel_data('rxnav_relations_with_extracted_diseases.xlsx')
    diseases = extractor.extract_unique_diseases(df)
    results = extractor.process_diseases(diseases)
    extractor.save_results(results, 'icd_codes.csv')

``icd_codes.csv`` will contain two columns: ``disease_name`` and
``icd_codes`` (a semicolon‑separated string of codes).  You can open
the CSV in any spreadsheet application for review.

Notes
-----
This script depends on external internet access to reach the CTSS API.
When running in restricted environments (e.g., inside some containers
or offline), the API calls may fail.  In such cases you can populate
the cache manually or by downloading an ICD‑10‑CM dataset and adapting
the ``search_icd_code`` method to use local data instead of the API.
"""

import argparse
import sys
import ast
import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd
import requests


def _safe_literal_eval(value: str) -> Optional[List[str]]:
    """Safely evaluate a string containing a Python list.

    Returns None if the string cannot be parsed.

    Parameters
    ----------
    value : str
        A string that should represent a Python list literal.

    Returns
    -------
    list[str] | None
        The parsed list of strings, or None if parsing fails.
    """
    if not isinstance(value, str):
        return None
    try:
        result = ast.literal_eval(value)
        # Ensure the result is a list of strings
        if isinstance(result, list):
            return [str(item).strip() for item in result]
    except (ValueError, SyntaxError):
        return None
    return None


@dataclass
class ICDExtractor:
    """Extracts ICD‑10 codes for disease names from an Excel file."""

    cache_file: str = 'icd_cache.json'
    base_url: str = 'https://clinicaltables.nlm.nih.gov/api/icd10cm/v3/search'
    cache: Dict[str, List[str]] = field(default_factory=dict)
    session: requests.Session = field(init=False)

    def __post_init__(self) -> None:
        # Initialize HTTP session and load cache
        self.session = requests.Session()
        self.cache = self.load_cache(self.cache_file)

    def load_cache(self, cache_file: str) -> Dict[str, List[str]]:
        """Load previously saved ICD code lookups from disk.

        Parameters
        ----------
        cache_file : str
            Path to a JSON file mapping disease names to lists of codes.

        Returns
        -------
        dict
            The loaded cache or an empty dictionary if the file does not exist.
        """
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (IOError, json.JSONDecodeError):
                # If the file is corrupt, start with an empty cache
                return {}
        return {}

    def save_cache(self) -> None:
        """Persist the current cache to disk."""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except IOError:
            pass  # Ignore failures to write the cache

    def load_excel_data(self, filepath: str) -> pd.DataFrame:
        """Load an Excel spreadsheet into a pandas DataFrame.

        Parameters
        ----------
        filepath : str
            Path to the Excel file.

        Returns
        -------
        pandas.DataFrame
            The loaded DataFrame.
        """
        return pd.read_excel(filepath)

    @staticmethod
    def _extract_from_cell(cell: str) -> List[str]:
        """Extract a list of diseases from a single cell.

        Cells are expected to contain a string representation of a Python list.
        Non‑list entries return an empty list.

        Parameters
        ----------
        cell : str
            The cell value.

        Returns
        -------
        list[str]
            A list of disease names.
        """
        diseases = _safe_literal_eval(cell)
        return diseases if diseases is not None else []

    def extract_unique_diseases(self, df: pd.DataFrame) -> Set[str]:
        """Extract a set of unique disease names from the specified columns.

        Parameters
        ----------
        df : pandas.DataFrame
            The DataFrame containing disease columns.

        Returns
        -------
        set[str]
            A set of unique disease names.
        """
        unique_diseases: Set[str] = set()
        # Columns to parse
        cols = [
            'may_prevent_diseases',
            'ci_with_diseases',
            'may_diagnose_diseases',
            'may_treat_diseases',
        ]
        for col in cols:
            if col not in df.columns:
                continue
            for cell in df[col].dropna():
                # Each cell is a string representation of a list
                diseases = self._extract_from_cell(cell)
                for dis in diseases:
                    normalized = dis.strip()
                    if normalized:
                        unique_diseases.add(normalized)
        return unique_diseases

    def _query_api(self, query: str, max_list: int = 20) -> Tuple[List[str], List[str]]:
        """Query the CTSS API for a given term.

        Returns a tuple of (codes, names).  Retries a few times on
        network errors.

        Parameters
        ----------
        query : str
            The term to search for.
        max_list : int, default=20
            Maximum number of results to return (API default is 7
           【125092994786881†L40-L60】).

        Returns
        -------
        tuple[list[str], list[str]]
            A tuple of (codes, names).  Either list may be empty if
            nothing was found or an error occurred.
        """
        params = {
            'sf': 'code,name',
            'terms': query,
            'maxList': str(max_list),
        }
        retries = 3
        backoff = 0.5
        for attempt in range(retries):
            try:
                resp = self.session.get(self.base_url, params=params, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    # The response format is: [total_count, [codes...], [extra], [display strings...], [codeSystem]]
                    codes = data[1] if len(data) > 1 else []
                    names = data[3] if len(data) > 3 else []
                    return codes, names
                else:
                    # Unexpected status; treat as no result
                    return [], []
            except (requests.RequestException, ValueError):
                # On network or parsing error, wait and retry
                time.sleep(backoff)
                backoff *= 2
        # If all attempts fail, return empty lists
        return [], []

    def search_icd_code(self, disease: str) -> List[str]:
        """Search for ICD‑10 codes corresponding to a disease name.

        This method first attempts an exact lookup.  If no codes
        are found, it falls back to splitting the disease on commas
        and whitespace to try simpler queries.

        Parameters
        ----------
        disease : str
            The disease name to search.

        Returns
        -------
        list[str]
            A list of ICD‑10 codes associated with the disease.
        """
        # Return cached result if available
        cached = self.cache.get(disease)
        if cached is not None:
            return cached

        codes: List[str] = []

        # First attempt: direct query
        result_codes, _ = self._query_api(disease)
        if result_codes:
            codes.extend(result_codes)
        else:
            # Fallback: split on comma
            parts = [p.strip() for p in re.split(r'[;,]', disease) if p.strip()]
            # Query each part separately
            for part in parts:
                tmp_codes, _ = self._query_api(part)
                if tmp_codes:
                    codes.extend(tmp_codes)
            # If still no codes, split on whitespace
            if not codes:
                tokens = [t.strip() for t in re.split(r'\s+', disease) if t.strip()]
                for token in tokens:
                    tmp_codes, _ = self._query_api(token)
                    if tmp_codes:
                        codes.extend(tmp_codes)
        # Remove duplicates while preserving order
        seen = set()
        unique_codes: List[str] = []
        for c in codes:
            if c not in seen:
                seen.add(c)
                unique_codes.append(c)
        # Update cache (store even if empty to avoid repeated lookups)
        self.cache[disease] = unique_codes
        return unique_codes

    def process_diseases(self, diseases: Iterable[str]) -> Dict[str, List[str]]:
        """Process an iterable of disease names and return a mapping to ICD codes.

        Parameters
        ----------
        diseases : iterable[str]
            Iterable of disease names.

        Returns
        -------
        dict[str, list[str]]
            Mapping from disease name to list of ICD codes.
        """
        results: Dict[str, List[str]] = {}
        for disease in diseases:
            codes = self.search_icd_code(disease)
            results[disease] = codes
        # Save cache after processing
        self.save_cache()
        return results

    def save_results(self, results: Dict[str, List[str]], output_file: str) -> None:
        """Save the disease–code mapping to a CSV or Excel file.

        The output contains two columns: ``disease_name`` and
        ``icd_codes``, where the codes are a semicolon‑separated string.

        Parameters
        ----------
        results : dict[str, list[str]]
            Mapping from disease name to codes.
        output_file : str
            Path to the output file.  The extension determines the
            format: ``.csv`` for comma‑separated values and ``.xlsx`` for
            Excel.
        """
        rows = []
        for disease, codes in results.items():
            rows.append({'disease_name': disease, 'icd_codes': ';'.join(codes)})
        out_df = pd.DataFrame(rows)
        if output_file.lower().endswith('.xlsx'):
            out_df.to_excel(output_file, index=False)
        else:
            out_df.to_csv(output_file, index=False)

    # ------------------------------------------------------------------
    # DataFrame augmentation
    #
    def add_icd_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of ``df`` with additional columns containing ICD‑10 codes.

        For each of the four disease columns (``may_prevent_diseases``,
        ``ci_with_diseases``, ``may_diagnose_diseases`` and
        ``may_treat_diseases``) this method creates a corresponding
        ``*_icd10`` column.  Each cell in the new column is a list of
        dictionaries where each dictionary has a single key–value pair
        mapping a disease name to the list of ICD codes returned by
        :meth:`search_icd_code`.  The structure of the new columns is
        therefore ``[{disease: [codes]}, ...]``.  Diseases that cannot be
        matched to any ICD codes will still be included in the list
        with an empty list of codes.

        Because lookups are cached, repeated diseases across rows
        incur no additional API cost.

        Parameters
        ----------
        df : pandas.DataFrame
            The DataFrame containing the original disease columns.

        Returns
        -------
        pandas.DataFrame
            A new DataFrame with four extra columns appended.
        """
        # Make a copy to avoid mutating the input DataFrame
        new_df = df.copy()
        cols = [
            'may_prevent_diseases',
            'ci_with_diseases',
            'may_diagnose_diseases',
            'may_treat_diseases',
        ]
        for col in cols:
            if col not in new_df.columns:
                continue
            new_col = []
            for cell in new_df[col].fillna('').astype(str):
                # parse diseases list from string
                diseases = self._extract_from_cell(cell)
                row_values: List[Dict[str, List[str]]] = []
                if diseases is None:
                    # ensure we append an empty list if parsing fails
                    new_col.append(row_values)
                    continue
                for dis in diseases:
                    disease_name = dis.strip()
                    if not disease_name:
                        continue
                    codes = self.search_icd_code(disease_name)
                    row_values.append({disease_name: codes})
                new_col.append(row_values)
            new_df[f"{col}_icd10"] = new_col
        return new_df

    def save_augmented(self, input_file: str, output_file: str) -> None:
        """Load an Excel file, add ICD code columns and save the result.

        This convenience method combines reading the input workbook,
        augmenting it with ICD‑10 columns (using :meth:`add_icd_columns`)
        and writing the augmented DataFrame to a file.  The output file
        format is determined by its extension (``.xlsx`` for Excel,
        otherwise CSV).

        Parameters
        ----------
        input_file : str
            Path to the input Excel file.
        output_file : str
            Path to the file to write the augmented DataFrame.  A
            ``.xlsx`` extension produces an Excel workbook; any other
            extension yields a CSV.
        """
        df = self.load_excel_data(input_file)
        augmented = self.add_icd_columns(df)
        # Save cache before writing results
        self.save_cache()
        if output_file.lower().endswith('.xlsx'):
            augmented.to_excel(output_file, index=False)
        else:
            augmented.to_csv(output_file, index=False)

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract ICD-10 codes for the extracted diseases."
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
    
    # Initialise the extractor (will load cache from icd_cache.json if it exists)
    extractor = ICDExtractor(cache_file='icd_cache.json')

    # Extract the set of unique disease names from the four disease columns
    disease_set = extractor.extract_unique_diseases(df_in)
    print(f'Found {len(disease_set)} unique diseases')

    try:
        # Query the API and build the mapping.  Results will be cached locally.
        df_out = extractor.process_diseases(disease_set)
    except Exception as e:
        print(f"Failed to process data: {e}", file=sys.stderr)
        sys.exit(1)
    try:
        extractor.save_results(df_out, 'data/output/icd_codes.csv')
        print(f"Wrote {len(df_out)} rows to {args.output}")
        extractor.save_augmented(
            input_file=args.input,
            output_file=args.output
        )
    except Exception as e:
        print(f"Failed to write output file: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()