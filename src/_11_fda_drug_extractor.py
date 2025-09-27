"""
fda_drug_extractor.py
=====================

This module defines the :class:`FDADrugExtractor`, a high‑level
interface for retrieving FDA approval information for a list of drug
names stored in an Excel file.  It automates query construction,
handles API rate limits and errors, caches results to avoid duplicate
requests, and augments the original dataset with structured approval
data.  The extractor relies on the U.S. Food and Drug Administration
openFDA API (`drugsfda` endpoint) to fetch approval histories.

The extractor supports intelligent search fallbacks for multi‑word
names, uses configurable parameters defined at the top of this module,
and produces a summary report describing processing outcomes.

Example
-------

```python
from fda_drug_extractor import FDADrugExtractor

extractor = FDADrugExtractor()
df = extractor.load_excel_data('rxnav_with_icd10.xlsx')
augmented = extractor.process_excel_drugs(df)
extractor.save_enhanced_results(augmented, 'rxnav_with_fda.xlsx')
```

See the accompanying README for detailed usage instructions and
configuration options.
"""

from __future__ import annotations

import argparse
import ast
import itertools
import json
import logging
import os
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd
import requests

try:  # Optional dependency for GPT assistance
    from openai import OpenAI
except ImportError:  # pragma: no cover - handled at runtime
    OpenAI = None
from requests import Response

# Configuration constants (formerly in fda_config)
# You can adjust these values to tune API behaviour and performance.
FDA_API_BASE_URL: str = "https://api.fda.gov/drug/drugsfda.json"
RATE_LIMIT_DELAY: float = 1.0  # seconds between API calls
MAX_RETRIES: int = 3           # maximum number of retries per request
TIMEOUT_SECONDS: int = 30      # HTTP timeout for API requests

# Type alias for the API result structure
FDAActionRow = Dict[str, Any]


def _iso(date_str: Optional[str]) -> Optional[str]:
    """Normalize dates from YYYYMMDD to YYYY‑MM‑DD.

    If the input is not a string of eight digits, return it unmodified.

    Parameters
    ----------
    date_str : str | None
        A string representing a date in YYYYMMDD format.

    Returns
    -------
    str | None
        A normalized date string (YYYY‑MM‑DD) or None.
    """
    if not date_str or not isinstance(date_str, str) or not date_str.isdigit() or len(date_str) != 8:
        return date_str
    return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"


class FDADrugExtractor:
    """Extract FDA approval data for drugs listed in an Excel file.

    The extractor reads an Excel workbook containing a column of drug
    names (default `'Drug Name'`), queries the openFDA `drugsfda` API
    to obtain approval histories, and appends structured data to the
    original DataFrame.  It supports intelligent search fallback
    strategies, rate limiting, request retries, caching, and a summary
    report.  Configuration parameters are defined in :mod:`fda_config`.
    """

    def __init__(self,
                 cache_file: str | None = None,
                 rate_limit_delay: float = RATE_LIMIT_DELAY,
                 max_retries: int = MAX_RETRIES,
                 timeout: int = TIMEOUT_SECONDS) -> None:
        # API endpoint
        self.base_url: str = FDA_API_BASE_URL
        # Rate limiting and retry configuration
        self.rate_limit_delay: float = rate_limit_delay
        self.max_retries: int = max_retries
        self.timeout: int = timeout
        # A session for HTTP requests to reuse connections
        self.session: requests.Session = requests.Session()
        # A cache mapping drug names to extracted data (list of dicts)
        self.drug_cache: Dict[str, List[FDAActionRow]] = {}
        # A set of search terms already queried to avoid duplicates
        self.processed_terms: Set[str] = set()
        # Path to persist cache between runs (optional)
        self.cache_file: str | None = cache_file
        if cache_file and os.path.exists(cache_file):
            self._load_cache()
        # Set up logging
        self.logger = self.setup_logging()

    # ------------------------------------------------------------------
    # Logging
    def setup_logging(self) -> logging.Logger:
        """Configure a logger for the extractor.

        Returns a logger instance configured to output messages to
        stderr with timestamps.  The log level is determined by the
        environment variable ``FDA_EXTRACTOR_LOGLEVEL`` (defaults to
        INFO).  Logging is essential for debugging and audit trails.
        """
        loglevel_str = os.environ.get('FDA_EXTRACTOR_LOGLEVEL', 'INFO').upper()
        loglevel = getattr(logging, loglevel_str, logging.INFO)
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        logger.setLevel(loglevel)
        return logger

    # ------------------------------------------------------------------
    # Cache handling
    def _load_cache(self) -> None:
        """Load cached drug results from the cache file into memory."""
        try:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                cached = json.load(f)
                # Convert keys to strings and values to list of dicts
                if isinstance(cached, dict):
                    for k, v in cached.items():
                        if isinstance(v, list):
                            self.drug_cache[k] = v
            self.logger.info(f"Loaded cache from {self.cache_file} ({len(self.drug_cache)} entries).")
        except (IOError, json.JSONDecodeError) as e:
            self.logger.warning(f"Failed to load cache: {e}")

    def _save_cache(self) -> None:
        """Persist the current cache to disk if a cache file is configured."""
        if not self.cache_file:
            return
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.drug_cache, f, ensure_ascii=False, indent=2)
            self.logger.info(f"Saved cache to {self.cache_file} ({len(self.drug_cache)} entries).")
        except IOError as e:
            self.logger.warning(f"Failed to save cache: {e}")

    # ------------------------------------------------------------------
    # Data loading
    def load_excel_data(self, filepath: str, sheet_name: str | int | None = 0) -> pd.DataFrame:
        """Read an Excel workbook and return a pandas DataFrame.

        Parameters
        ----------
        filepath : str
            Path to the Excel file (.xlsx) to read.
        sheet_name : str or None, default None
            Name or index of the sheet to read.  If None, the first
            sheet is used.

        Returns
        -------
        pandas.DataFrame
            The loaded data.

        Raises
        ------
        ValueError
            If the file cannot be read or the expected column is missing.
        """
        # When sheet_name=None, pandas returns a dict of DataFrames for all sheets.
        # Default to sheet 0 (first sheet) to ensure a single DataFrame is returned.
        try:
            df = pd.read_excel(filepath, sheet_name=sheet_name)
        except Exception as e:
            raise ValueError(f"Failed to read Excel file {filepath}: {e}") from e
        # If a dict is returned (multiple sheets), select the first sheet
        if isinstance(df, dict):
            # Take the first sheet's DataFrame
            first_key = next(iter(df))
            df = df[first_key]
        # Ensure the 'Drug Name' column exists
        if 'Drug Name' not in df.columns:
            raise ValueError(f"Expected 'Drug Name' column in {filepath}.")
        return df


    # ------------------------------------------------------------------
    # Search term generation
    def generate_search_combinations(self, drug_name: str) -> List[str]:
        """Generate a prioritized list of search terms for a drug name.

        The strategy includes exact matches, variants without common
        pharmaceutical suffixes, individual significant words,
        combinations of adjacent words, and alternative separators.  The
        resulting list preserves order of importance.

        Parameters
        ----------
        drug_name : str
            The original drug name from the dataset.

        Returns
        -------
        list[str]
            A list of candidate search terms.
        """
        name = (drug_name or '').strip()
        if not name:
            return []
        words = name.split()
        combos: List[str] = []
        # Priority 1: exact match as provided
        combos.append(name)
        # Priority 2: remove common pharmaceutical suffixes from the last word
        pharma_suffixes = ['XR', 'ER', 'SR', 'CR', 'LA', 'XL', 'ODT']
        def strip_suffix(w: str) -> str:
            for suf in pharma_suffixes:
                if w.upper().endswith(suf) and len(w) > len(suf):
                    return w[:-len(suf)]
            return w
        # If the name has multiple words, try stripping suffix from last word
        if len(words) > 1:
            stripped_last = strip_suffix(words[-1])
            if stripped_last != words[-1]:
                combos.append(' '.join(words[:-1] + [stripped_last]))
        # Priority 3: individual significant words (length > 2)
        common_stopwords: Set[str] = {'and', 'or', 'with', 'the', 'for'}
        significant_words = [w for w in words if len(w) > 2 and w.lower() not in common_stopwords]
        combos.extend(significant_words)
        # Priority 4: generate combinations of consecutive words (2‑word, 3‑word, etc.)
        # Only consider combinations that preserve order of words
        n = len(words)
        # Consider 2-word combinations
        for i in range(n):
            for j in range(i + 1, min(i + 3, n)):  # up to pairs/triples
                combo_words = words[i:j + 1]
                term = ' '.join(combo_words)
                if term not in combos:
                    combos.append(term)
        # Priority 5: produce plus-separated variants for multi-word combinations
        for term in list(combos):
            if ' ' in term:
                plus_term = term.replace(' ', '+')
                if plus_term not in combos:
                    combos.append(plus_term)
        # Deduplicate while preserving order
        seen: Set[str] = set()
        deduped: List[str] = []
        for term in combos:
            if term not in seen:
                seen.add(term)
                deduped.append(term)
        return deduped

    # ------------------------------------------------------------------
    # API calling
    def search_fda_api(self, search_term: str) -> List[Dict[str, Any]]:
        """Query the openFDA API for a given search term.

        This method respects rate limits and retry policies.  It
        automatically interprets HTTP status codes (404 indicates no
        results) and handles 429 (Too Many Requests) by delaying.

        Parameters
        ----------
        search_term : str
            A term to include in the API `search` parameter.  The term
            should not be URL-encoded; quoting and encoding are handled
            by requests.

        Returns
        -------
        list[dict]
            The `results` array from the API response, or an empty
            list if no results were found.

        Raises
        ------
        RuntimeError
            If all retries fail due to network or server errors.
        """
        # The API expects a query like 'products.brand_name:"ADVIL PM"'
        params = {
            'search': f'products.brand_name:"{search_term}"',
            'limit': '1'
        }
        attempt = 0
        while attempt < self.max_retries:
            try:
                response: Response = self.session.get(
                    self.base_url, params=params, timeout=self.timeout
                )
                status = response.status_code
                # 404 means no results for this search term
                if status == 404:
                    return []
                # 429 indicates rate limiting; respect Retry-After header if present
                if status == 429:
                    retry_after = response.headers.get('Retry-After')
                    try:
                        wait = int(retry_after) if retry_after else None
                    except ValueError:
                        wait = None
                    sleep_time = wait if wait is not None else min(300, 2 ** attempt)
                    self.logger.debug(f"Rate limited on term '{search_term}'; sleeping for {sleep_time}s.")
                    time.sleep(sleep_time)
                    attempt += 1
                    continue
                # 5xx server errors
                if 500 <= status < 600:
                    self.logger.debug(f"Server error {status} on term '{search_term}'; retrying.")
                    time.sleep(min(60, 2 ** attempt))
                    attempt += 1
                    continue
                # Raise for other client errors (400 etc.)
                response.raise_for_status()
                data = response.json()
                results = data.get('results', [])
                return results
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                self.logger.debug(f"Network error on term '{search_term}': {e}; retrying.")
                time.sleep(min(60, 2 ** attempt))
                attempt += 1
                continue
            except requests.exceptions.HTTPError as e:
                # Unexpected client error; abort
                self.logger.warning(f"HTTP error on term '{search_term}': {e}")
                raise
        # If we exhaust retries
        raise RuntimeError(f"Failed to fetch data for term '{search_term}' after {self.max_retries} attempts.")

    # ------------------------------------------------------------------
    # Flatten actions
    def _flatten_actions(self, results: Iterable[Dict[str, Any]]) -> List[FDAActionRow]:
        """Transform raw API results into structured action rows.

        For each submission in each returned application, produce
        dictionaries containing key attributes and a derived action
        type/date if the `actions` array is missing.  The output
        structure matches the example provided in the task description.

        Parameters
        ----------
        results : iterable of dict
            The list of application objects returned from the API.

        Returns
        -------
        list[dict]
            A list of flattened action rows.
        """
        out: List[FDAActionRow] = []
        for app in results:
            app_no = app.get('application_number')
            products = app.get('products', []) or []
            brand_names = self._collect_brand_names(products)
            generic_names = self._collect_generic_names(products)
            for sub in app.get('submissions', []) or []:
                base = {
                    'application_number': app_no,
                    'brand_names': brand_names,
                    'generic_names': generic_names,
                    'submission_type': sub.get('submission_type'),
                    'submission_number': sub.get('submission_number'),
                    'submission_status': sub.get('submission_status'),
                    'submission_status_date': _iso(sub.get('submission_status_date')),
                    'submission_class_code': sub.get('submission_class_code'),
                    'submission_class_code_description': sub.get('submission_class_code_description'),
                    'review_priority': sub.get('review_priority'),
                    'docs': [
                        {
                            'type': d.get('type'),
                            'date': _iso(d.get('date')),
                            'url': d.get('url'),
                            'id': d.get('id'),
                        }
                        for d in (sub.get('application_docs') or [])
                    ],
                }
                # Use actions if present
                acts = sub.get('actions') or []
                if acts:
                    any_row = False
                    for a in acts:
                        atype = a.get('action_type')
                        adate = _iso(a.get('action_date'))
                        if atype or adate:
                            any_row = True
                            out.append({**base, 'action_type': atype, 'action_date': adate})
                    if not any_row:
                        # derive fallback
                        out.append({**base,
                                    'action_type': self._derive_action_type(sub),
                                    'action_date': self._best_submission_date(sub)})
                else:
                    # derive fallback
                    out.append({**base,
                                'action_type': self._derive_action_type(sub),
                                'action_date': self._best_submission_date(sub)})
        return out

    # Helpers for flatten actions
    def _derive_action_type(self, sub: Dict[str, Any]) -> str:
        stype = (sub.get('submission_type') or '').upper()
        status = (sub.get('submission_status') or '').upper()
        desc = sub.get('submission_class_code_description') or sub.get('submission_class_code') or ''
        if stype == 'ORIG' and status == 'AP':
            return 'Approval'
        if stype == 'SUPPL' and status == 'AP':
            return f"Supplement - {desc}" if desc else 'Supplement'
        return f"Submission - {stype or 'UNKNOWN'}"

    def _best_submission_date(self, sub: Dict[str, Any]) -> Optional[str]:
        ssd = _iso(sub.get('submission_status_date'))
        if ssd:
            return ssd
        docs = sub.get('application_docs') or []
        dates = [d for d in (doc.get('date') for doc in docs) if d]
        return _iso(sorted(dates)[-1]) if dates else None

    def _collect_brand_names(self, products: List[Dict[str, Any]]) -> str:
        brands = sorted({p.get('brand_name') for p in (products or []) if p.get('brand_name')})
        return '; '.join(brands)

    def _collect_generic_names(self, products: List[Dict[str, Any]]) -> str:
        gens: Set[str] = set()
        for p in (products or []):
            # Use generic_name if present; else active ingredients
            if p.get('generic_name'):
                gens.add(p['generic_name'])
            for ai in p.get('active_ingredients') or []:
                if isinstance(ai, dict) and ai.get('name'):
                    gens.add(ai['name'])
        return '; '.join(sorted(gens))

    # ------------------------------------------------------------------
    # Data extraction for single drug
    def extract_drug_data(self, drug_name: str) -> Tuple[List[FDAActionRow], bool, List[str], int]:
        """Retrieve approval data for a single drug using fallback search.

        Attempts to find matching approval records for the provided drug
        name by trying a sequence of search terms generated by
        :meth:`generate_search_combinations`.  Stops at the first
        successful search (non‑empty results) and caches the result.

        Parameters
        ----------
        drug_name : str
            The name of the drug to search.

        Returns
        -------
        tuple
            A four‑element tuple: (data, success_flag, terms_tried,
            api_calls).  ``data`` is a list of action row dicts;
            ``success_flag`` indicates whether any match was found;
            ``terms_tried`` is the ordered list of search terms used;
            ``api_calls`` is the number of API requests made for this
            drug.
        """
        # Check cache
        if drug_name in self.drug_cache:
            return self.drug_cache[drug_name], bool(self.drug_cache[drug_name]), [], 0
        terms = self.generate_search_combinations(drug_name)
        api_calls = 0
        for term in terms:
            if term in self.processed_terms:
                # skip previously processed terms to avoid duplicate API calls
                continue
            # Respect global rate limit
            if api_calls > 0:
                time.sleep(self.rate_limit_delay)
            # Query API
            try:
                results = self.search_fda_api(term)
                api_calls += 1
                self.processed_terms.add(term)
            except Exception as e:
                # Log and move to next term
                self.logger.warning(f"Error searching term '{term}' for '{drug_name}': {e}")
                continue
            # Flatten results
            if results:
                data_rows = self._flatten_actions(results)
                # Cache and return
                self.drug_cache[drug_name] = data_rows
                return data_rows, True, terms[:terms.index(term) + 1], api_calls
        # If no term yielded results, cache empty
        self.drug_cache[drug_name] = []
        return [], False, terms, api_calls

    # ------------------------------------------------------------------
    # Batch processing
    def process_excel_drugs(self, df: pd.DataFrame, drug_column: str = 'Drug Name') -> pd.DataFrame:
        """Process all drugs in the DataFrame and augment with approval data.

        Parameters
        ----------
        df : pandas.DataFrame
            The original DataFrame containing at least the specified
            drug column.
        drug_column : str, default 'Drug Name'
            Name of the column containing drug names.

        Returns
        -------
        pandas.DataFrame
            A copy of the original DataFrame with additional columns:
            ``fda_approval_data`` (JSON string of action rows),
            ``search_successful`` (bool), ``search_terms_attempted``
            (list of terms), ``api_calls_made`` (int), and
            ``last_updated`` (timestamp of extraction).
        """
        if drug_column not in df.columns:
            raise ValueError(f"Column '{drug_column}' not found in DataFrame.")
        # Prepare output columns
        augmented = df.copy()
        augmented['fda_approval_data'] = None
        augmented['search_successful'] = False
        augmented['search_terms_attempted'] = None
        augmented['api_calls_made'] = 0
        augmented['last_updated'] = None
        start_time = time.time()
        # Use tqdm if available for progress bar
        try:
            from tqdm import tqdm
            iter_range = tqdm(range(len(augmented)), desc='Processing drugs')
        except ImportError:
            iter_range = range(len(augmented))
        for idx in iter_range:
            drug_name = str(augmented.at[idx, drug_column]).strip()
            if not drug_name:
                augmented.at[idx, 'fda_approval_data'] = json.dumps([])
                augmented.at[idx, 'search_successful'] = False
                augmented.at[idx, 'search_terms_attempted'] = []
                augmented.at[idx, 'api_calls_made'] = 0
                augmented.at[idx, 'last_updated'] = time.strftime('%Y-%m-%d %H:%M:%S')
                continue
            data, success, terms_used, calls = self.extract_drug_data(drug_name)
            # Serialize data to JSON string to ensure Excel friendliness
            augmented.at[idx, 'fda_approval_data'] = json.dumps(data, ensure_ascii=False)
            augmented.at[idx, 'search_successful'] = success
            augmented.at[idx, 'search_terms_attempted'] = terms_used
            augmented.at[idx, 'api_calls_made'] = calls
            augmented.at[idx, 'last_updated'] = time.strftime('%Y-%m-%d %H:%M:%S')
        # Persist cache after processing
        self._save_cache()
        end_time = time.time()
        self.logger.info(f"Processed {len(augmented)} drugs in {end_time - start_time:.2f} seconds.")
        return augmented

    # ------------------------------------------------------------------
    # Save results
    def save_enhanced_results(self, df: pd.DataFrame, output_path: str) -> None:
        """Save the augmented DataFrame to a file.

        Supports writing to Excel (.xlsx) or CSV.  The file format is
        inferred from the extension.  Raises an error for unsupported
        formats.

        Parameters
        ----------
        df : pandas.DataFrame
            The augmented DataFrame returned by
            :meth:`process_excel_drugs`.
        output_path : str
            Destination path for the output file.
        """
        ext = os.path.splitext(output_path)[1].lower()
        if ext == '.xlsx':
            df.to_excel(output_path, index=False)
        elif ext == '.csv':
            df.to_csv(output_path, index=False)
        else:
            raise ValueError(f"Unsupported output format: {ext}")
        self.logger.info(f"Saved enhanced results to {output_path}.")

    # ------------------------------------------------------------------
    # Summary
    def generate_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate a summary report from the augmented DataFrame.

        Parameters
        ----------
        df : pandas.DataFrame
            The DataFrame returned by :meth:`process_excel_drugs`.

        Returns
        -------
        dict
            A summary dictionary containing totals, success rates,
            total API calls, and processing time if available.
        """
        total = len(df)
        successes = int(df['search_successful'].sum())
        failures = total - successes
        total_calls = int(df['api_calls_made'].sum())
        summary = {
            'total_drugs_processed': total,
            'successful_matches': successes,
            'failed_searches': failures,
            'total_api_calls': total_calls,
        }
        return summary
