import numpy as np
import pandas as pd
import re
import ast
import argparse
import os
from collections import defaultdict

def merge_dicts_from_string(s):
    """
    Convert a string representation of a list of dicts into
    a single merged dict with deduplicated values.
    """
    if pd.isna(s) or not s.strip():
        return {}
    
    try:
        parsed = ast.literal_eval(s)  # safe parse
    except Exception:
        return {}
    
    merged = defaultdict(list)
    for d in parsed:
        for k, v in d.items():
            for code in v:
                if code not in merged[k]:
                    merged[k].append(code)  # dedup + preserve order
    return dict(merged)


def extract_icd10_prefix_by_key(d):
    """
    Given a dict like {'Pain': ['R07.2', 'R07.82'], 'Anxiety': ['F41.9']},
    return {'Pain': ['R07'], 'Anxiety': ['F41']}.
    """
    if not isinstance(d, dict):
        return {}

    result = {}
    for k, codes in d.items():
        prefixes = set()
        for code in codes:
            prefixes.add(code.split(".")[0])  # take prefix before "."
        result[k] = sorted(prefixes)  # keep unique & sorted
    return result


def _normalize_prefix_dict(d):
    """
    Ensure each key maps to a unique, ordered list of non-empty string prefixes.
    """
    if not isinstance(d, dict):
        return {}

    out = {}
    for k, vals in d.items():
        seen = set()
        cleaned = []
        for v in vals or []:
            if not isinstance(v, str):
                continue
            v = v.strip().upper()
            if not v:
                continue
            if v not in seen:
                seen.add(v)
                cleaned.append(v)
        out[k] = cleaned
    return out

def combine_continuous_icd10(d):
    """
    For each key, merge consecutive ICD-10 prefixes with the same letter(s):
      F10, F11, F12 -> F10-F12
    Non-consecutive codes remain as-is (e.g., T40, Z99).
    """
    if not isinstance(d, dict):
        return {}

    result = {}
    for k, codes in d.items():
        # parse (letters, number) for sortable grouping; keep unknowns as passthrough
        parsed = []
        passthrough = []
        for code in codes:
            m = re.match(r"^([A-Z]+)(\d+)$", code)
            if m:
                letter, num = m.groups()
                parsed.append((letter, int(num)))
            else:
                # If it doesn't match LETTER+NUMBER (e.g., 'U07', 'X99A'), keep as-is
                passthrough.append(code)

        # sort by (letter, number)
        parsed.sort(key=lambda x: (x[0], x[1]))

        compressed = []
        i = 0
        while i < len(parsed):
            letter, start_num = parsed[i]
            end_num = start_num
            j = i + 1
            # grow range while consecutive within same letter group
            while j < len(parsed) and parsed[j][0] == letter and parsed[j][1] == end_num + 1:
                end_num = parsed[j][1]
                j += 1

            if start_num == end_num:
                compressed.append(f"{letter}{start_num}")
            else:
                compressed.append(f"{letter}{start_num}-{letter}{end_num}")

            i = j

        # keep original non-LETTER+NUMBER items in their original relative order at the end
        result[k] = compressed + passthrough

    return result


def main():
    """Main function to process the knowledge graph preprocessing."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Preprocess pharmaceutical data for knowledge graph construction"
    )
    parser.add_argument(
        "--input", 
        required=True, 
        help="Path to input Excel file (e.g., rxnav_with_fda_with_orig2.xlsx)"
    )
    parser.add_argument(
        "--output_dir", 
        required=True, 
        help="Directory to save the processed output file"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load the data
    print(f"Loading data from: {args.input}")
    df = pd.read_excel(args.input)
    print(f"Loaded {len(df)} rows")
    
    # Apply preprocessing functions
    print("Preprocessing ICD-10 disease mappings...")
    df["may_treat_diseases_icd10_normalized"] = df["may_treat_diseases_icd10"].apply(merge_dicts_from_string)
    df["may_treat_diseases_icd10_normalized"] = df["may_treat_diseases_icd10_normalized"].apply(extract_icd10_prefix_by_key)
    df["may_treat_diseases_icd10_normalized"] = (
        df["may_treat_diseases_icd10_normalized"]
          .apply(_normalize_prefix_dict)
          .apply(combine_continuous_icd10)
    )
    
    # Save the processed data
    output_file = os.path.join(args.output_dir, "rxnav_with_kg.xlsx")
    print(f"Saving processed data to: {output_file}")
    df.to_excel(output_file, index=False)
    print("Preprocessing completed successfully!")


if __name__ == "__main__":
    main()