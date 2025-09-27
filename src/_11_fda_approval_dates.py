#!/usr/bin/env python3
import argparse
import json
import time
from typing import List, Dict, Any, Iterable, Tuple, Optional
import requests

OPENFDA_URL = "https://api.fda.gov/drug/drugsfda.json"

# ----------------------------
# Utilities
# ----------------------------
def _iso(d: Optional[str]) -> Optional[str]:
    """Convert YYYYMMDD -> YYYY-MM-DD if possible; otherwise return as-is."""
    if not d or not isinstance(d, str) or not d.isdigit() or len(d) != 8:
        return d
    return f"{d[:4]}-{d[4:6]}-{d[6:8]}"

def _sleep_backoff(attempt: int, base: float = 0.75, cap: float = 6.0):
    time.sleep(min(cap, base * (2 ** attempt)))

# ----------------------------
# HTTP + Error handling
# ----------------------------
def fetch_drugsfda(search: str, limit: int = 1, max_retries: int = 3) -> List[Dict[str, Any]]:
    """
    Call openFDA /drug/drugsfda with retries and friendly handling:
      - 404 => [] (openFDA uses 404 for 'no results')
      - 429 => honor Retry-After if present; else backoff and retry
      - 5xx / timeouts / connection errors => backoff and retry
      - other 4xx => raise
    """
    params = {"search": search, "limit": limit}
    last_err = None
    for attempt in range(max_retries):
        try:
            r = requests.get(OPENFDA_URL, params=params, timeout=30)
            if r.status_code == 404:
                return []
            if r.status_code == 429:
                ra = r.headers.get("Retry-After")
                try:
                    wait_s = int(ra) if ra else None
                except Exception:
                    wait_s = None
                time.sleep(wait_s if wait_s is not None else min(6, 0.75 * (2 ** attempt)))
                continue
            if 500 <= r.status_code < 600:
                _sleep_backoff(attempt); continue
            r.raise_for_status()
            return r.json().get("results", [])
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            last_err = e; _sleep_backoff(attempt); continue
        except requests.exceptions.HTTPError:
            raise
    if last_err:
        raise RuntimeError(f"openFDA request failed after {max_retries} attempts: {last_err}") from last_err
    return []

# ----------------------------
# Action derivation (for payloads WITHOUT actions[])
# ----------------------------
def _derive_action_type(sub: Dict[str, Any]) -> str:
    stype = (sub.get("submission_type") or "").upper()
    status = (sub.get("submission_status") or "").upper()
    desc = sub.get("submission_class_code_description") or sub.get("submission_class_code") or ""
    if stype == "ORIG" and status == "AP":
        return "Approval"
    if stype == "SUPPL" and status == "AP":
        return f"Supplement - {desc}" if desc else "Supplement"
    return f"Submission - {stype or 'UNKNOWN'}"

def _best_submission_date(sub: Dict[str, Any]) -> Optional[str]:
    """Pick an action_date: submission_status_date first, else latest doc.date."""
    ssd = _iso(sub.get("submission_status_date"))
    if ssd:
        return ssd
    docs = sub.get("application_docs") or []
    dates = [d for d in (doc.get("date") for doc in docs) if d]
    return _iso(sorted(dates)[-1]) if dates else None

# ----------------------------
# Transform — supports BOTH schemas
# ----------------------------
def _collect_brand_names(products: List[Dict[str, Any]]) -> str:
    brands = sorted({p.get("brand_name") for p in (products or []) if p.get("brand_name")})
    return "; ".join(brands)

def _collect_generic_names(products: List[Dict[str, Any]]) -> str:
    gens = set()
    for p in (products or []):
        # Prefer openfda.generic_name if present; otherwise fall back to active_ingredients[].name
        if p.get("generic_name"):
            gens.add(p["generic_name"])
        for ai in p.get("active_ingredients") or []:
            if isinstance(ai, dict) and ai.get("name"):
                gens.add(ai["name"])
    return "; ".join(sorted(gens))

def flatten_actions(results: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Build action dictionaries from either:
      - submissions[].actions[] (older/other apps), OR
      - derived from submission fields if actions[] missing (newer apps like KISUNLA).
    Returns: List[Dict] with keys:
      application_number, brand_names, generic_names, submission_type, submission_number,
      submission_status, submission_status_date, submission_class_code, submission_class_code_description,
      review_priority, action_type, action_date, docs (list of {type, date, url, id})
    """
    out: List[Dict[str, Any]] = []
    for app in results:
        app_no = app.get("application_number")  # e.g., "BLA761248"
        products = app.get("products", []) or []
        brand_names = _collect_brand_names(products)
        generic_names = _collect_generic_names(products)

        for sub in app.get("submissions", []) or []:
            base = {
                "application_number": app_no,
                "brand_names": brand_names,
                "generic_names": generic_names,
                "submission_type": sub.get("submission_type"),
                "submission_number": sub.get("submission_number"),
                "submission_status": sub.get("submission_status"),
                "submission_status_date": _iso(sub.get("submission_status_date")),
                "submission_class_code": sub.get("submission_class_code"),
                "submission_class_code_description": sub.get("submission_class_code_description"),
                "review_priority": sub.get("review_priority"),
                "docs": [
                    {
                        "type": d.get("type"),
                        "date": _iso(d.get("date")),
                        "url": d.get("url"),
                        "id": d.get("id"),
                    }
                    for d in (sub.get("application_docs") or [])
                ],
            }

            acts = sub.get("actions") or []
            if acts:
                any_row = False
                for a in acts:
                    atype = a.get("action_type")
                    adate = _iso(a.get("action_date"))
                    if atype or adate:
                        any_row = True
                        out.append({**base, "action_type": atype, "action_date": adate})
                if not any_row:
                    out.append({**base,
                                "action_type": _derive_action_type(sub),
                                "action_date": _best_submission_date(sub)})
            else:
                out.append({**base,
                            "action_type": _derive_action_type(sub),
                            "action_date": _best_submission_date(sub)})
    return out

# ----------------------------
# Public API: return List[Dict]
# ----------------------------
def get_openfda_actions(*, brand: Optional[str] = None, app: Optional[str] = None, limit: int = 1) -> List[Dict[str, Any]]:
    """
    Fetch and return action-like rows as a list of dictionaries.
    - brand: brand name (e.g., "KISUNLA")
    - app: application number (either "761248" or "BLA761248")
    - limit: number of application records to request (usually 1)
    """
    if not brand and not app:
        raise ValueError("Provide at least one of brand or app")

    rows: List[Dict[str, Any]] = []

    # Query by application number first (if provided)
    if app:
        q = f'application_number:"BLA{app}"' if app.isdigit() else f'application_number:"{app}"'
        rows.extend(flatten_actions(fetch_drugsfda(q, limit=limit)))

    # Also query by brand (if provided)
    if brand:
        rows.extend(flatten_actions(fetch_drugsfda(f'products.brand_name:"{brand}"', limit=limit)))

    # Deduplicate per submission + derived action
    def _row_key(r: Dict[str, Any]) -> Tuple:
        return (
            r.get("application_number"),
            r.get("submission_type"),
            r.get("submission_number"),
            r.get("action_type"),
            r.get("action_date"),
        )
    unique = { _row_key(r): r for r in rows }
    rows = list(unique.values())

    # Sort by action_date (None last), then action_type
    rows.sort(key=lambda r: ((r.get("action_date") or "9999-99-99"), r.get("action_type") or ""))
    return rows

# ----------------------------
# CLI for convenience (prints JSON)
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Return action-like rows (list of dicts) from openFDA /drug/drugsfda.")
    parser.add_argument("--brand", help='Brand name, e.g. "KISUNLA"')
    parser.add_argument("--app", help='Application number: "761248" or "BLA761248"')
    parser.add_argument("--limit", type=int, default=1, help="Records to request (default: 1)")
    args = parser.parse_args()

    data = get_openfda_actions(brand=args.brand, app=args.app, limit=args.limit)
    print(json.dumps(data, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
