#!/usr/bin/env python3
"""
Extract original FDA approval details from an Excel column.

Usage:
  python extract_fda_original.py \
      --input rxnav_with_fda.xlsx \
      --output rxnav_with_fda_with_orig.xlsx \
      --column fda_approval_data
"""

import argparse
import json
import ast
from datetime import date
from typing import Any, Dict, List, Optional, Union

import pandas as pd


def _to_date(s: Optional[str]) -> Optional[date]:
    if not s or not isinstance(s, str):
        return None
    try:
        return date.fromisoformat(s)
    except Exception:
        return None


def parse_fda_blob(fda_raw: Union[str, List[Dict[str, Any]], None]) -> Optional[List[Dict[str, Any]]]:
    """Returns a list of dicts or None."""
    if fda_raw is None or fda_raw == "" or (isinstance(fda_raw, float) and pd.isna(fda_raw)):
        return None
    if isinstance(fda_raw, list):
        return fda_raw
    if isinstance(fda_raw, str):
        # Try JSON first; fall back to Python literal (single quotes)
        try:
            return json.loads(fda_raw)
        except Exception:
            try:
                parsed = ast.literal_eval(fda_raw)
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                return None
    return None


def extract_original_approval(fda_raw: Union[str, List[Dict[str, Any]], None]) -> Optional[Dict[str, Any]]:
    """
    Returns a dict with original approval info, or None if not found.
    Priority:
      - submission_type == "ORIG"
      - among those, prefer action_type == "Approval"
      - else any ORIG with submission_status == "AP"
    For date, prefer action_date, otherwise submission_status_date.
    """
    records = parse_fda_blob(fda_raw)
    if not records:
        return None

    orig = [r for r in records if r.get("submission_type") == "ORIG"]
    if not orig:
        return None

    approved = [r for r in orig if str(r.get("action_type", "")).lower() == "approval"]
    if not approved:
        approved = [r for r in orig if r.get("submission_status") == "AP"]
    if not approved:
        return None

    def when(r: Dict[str, Any]) -> date:
        return _to_date(r.get("action_date")) or _to_date(r.get("submission_status_date")) or date.max

    best = min(approved, key=when)
    approval_date = best.get("action_date") or best.get("submission_status_date")

    return {
        "application_number": best.get("application_number"),
        "brand_names": best.get("brand_names"),
        "generic_names": best.get("generic_names"),
        "submission_type": best.get("submission_type"),
        "submission_number": best.get("submission_number"),
        "submission_status": best.get("submission_status"),
        "approval_date": approval_date,
    }


def main():
    ap = argparse.ArgumentParser(description="Extract original FDA approval info from an Excel column.")
    ap.add_argument("--input", required=True, help="Input .xlsx file path")
    ap.add_argument("--output", required=True, help="Output .xlsx file path")
    ap.add_argument("--sheet", default=None, help="Sheet name to read (default: first sheet)")
    ap.add_argument("--column", default="fda_approval_data", help="Column containing FDA data (default: fda_approval_data)")
    ap.add_argument("--output-sheet", default="Sheet1", help="Sheet name for output Excel (default: Sheet1)")
    args = ap.parse_args()

    # Load
    df = pd.read_excel(args.input, sheet_name=args.sheet or 0)

    if args.column not in df.columns:
        raise ValueError(f"Column '{args.column}' not found in the input file.")

    # Apply extraction
    extracted = df[args.column].apply(extract_original_approval)

    # Expand into new columns
    df["orig_application_number"] = extracted.apply(lambda x: x.get("application_number") if isinstance(x, dict) else None)
    df["orig_brand_names"]       = extracted.apply(lambda x: x.get("brand_names") if isinstance(x, dict) else None)
    df["orig_generic_names"]     = extracted.apply(lambda x: x.get("generic_names") if isinstance(x, dict) else None)
    df["orig_submission_type"]   = extracted.apply(lambda x: x.get("submission_type") if isinstance(x, dict) else None)
    df["orig_submission_number"] = extracted.apply(lambda x: x.get("submission_number") if isinstance(x, dict) else None)
    df["orig_submission_status"] = extracted.apply(lambda x: x.get("submission_status") if isinstance(x, dict) else None)
    df["orig_approval_date"]     = extracted.apply(lambda x: x.get("approval_date") if isinstance(x, dict) else None)

    # Save
    with pd.ExcelWriter(args.output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=args.output_sheet)

    # Simple summary
    total = len(df)
    found = df["orig_application_number"].notna().sum()
    print(f"Rows processed: {total} | Original approvals found: {found}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
