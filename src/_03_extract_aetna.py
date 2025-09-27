"""
Extract drug details from the Aetna formulary PDF.

This script is tailored to the Aetna formulary that lists drugs from pages 8–73
(1‑indexed). Each page contains a single table without explicit lines
separating rows or columns. The columns are conceptually:

    • **Drug Name** – The medication and formulation.
    • **Drug Tier** – A single digit (1–5) representing the coverage tier.
    • **Requirements/Limits** – Utilization management codes such as QL, PA, MO.

Drug family or category headings (e.g. “ANALGESICS”, “NSAIDS”) appear as
standalone lines in uppercase and apply to the subsequent drug rows until
another heading appears. The absence of ruling lines requires parsing based
on the spatial positions of the words. The algorithm proceeds as follows:

1. **Word Extraction**: Use pdfplumber to extract words with their
   coordinates for each page.
2. **Line Grouping**: Group words into lines by their y‑coordinate.
3. **Tier Column Detection**: Identify the x‑coordinate range of the tier
   column by clustering single‑digit tokens. This helps distinguish tier
   numbers from digits in drug descriptions.
4. **Row Reconstruction**: For each line within the table region:
   - Skip header lines containing “Drug Name”, “Tier” or “Requirements/Limits”.
   - If the line contains no digits, treat it as a new family heading if
     it is uppercase, otherwise treat it as a continuation line.
   - If the line contains a tier token within the detected tier column,
     start a new drug row. Words to the left form the drug name; words to
     the right form the requirements/limits.
   - Lines with digits but no tier token are treated as continuations of the
     current row.
5. **Output**: Accumulate rows with fields: family, name, tier, req and
   write them to an Excel file.

Run this script via:

    python extract_aetna_drug_details.py --input Aetna.pdf --output aetna_formulary.xlsx

Dependencies: pdfplumber, xlsxwriter
"""

import argparse
import re
from collections import defaultdict
from typing import List, Dict, Tuple

import pdfplumber
import xlsxwriter


def compute_tier_bounds(words: List[Dict], page_width: float) -> Tuple[float, float]:
    """Determine the x‑coordinate bounds of the tier column.

    Clusters the x0 positions of single‑digit words to find the most
    frequently occurring range, which corresponds to the tier column. A
    fallback range is returned if no single‑digit tokens are present.

    Args:
        words: List of word dicts from pdfplumber.extract_words().
        page_width: Width of the PDF page.

    Returns:
        (tier_left, tier_right): approximate bounds of the tier column.
    """
    digit_words = [w for w in words if w["text"].strip().isdigit() and len(w["text"].strip()) == 1]
    if not digit_words:
        # Use a generic right‑side range if no digits are found
        return page_width * 0.5, page_width * 0.7
    clusters: Dict[int, List[Dict]] = defaultdict(list)
    for w in digit_words:
        key = round(w["x0"] / 10) * 10
        clusters[key].append(w)
    best_cluster = max(clusters.values(), key=len)
    tier_left = min(w["x0"] for w in best_cluster) - 2
    tier_right = max(w["x1"] for w in best_cluster) + 2
    return tier_left, tier_right


def parse_page(page: pdfplumber.page.Page, current_family: str) -> Tuple[List[Dict], str]:
    """Parse a single page of the Aetna formulary to extract drug rows.

    Args:
        page: A pdfplumber Page object.
        current_family: The drug family carried over from previous page.

    Returns:
        (rows, last_family): A tuple where `rows` is a list of dictionaries
        with keys ``family``, ``name``, ``tier``, and ``req``. ``last_family``
        is the most recent family name encountered, to be passed to the next page.
    """
    words = page.extract_words(x_tolerance=1, y_tolerance=1)
    if not words:
        return [], current_family
    # Group words into lines by similar y coordinate
    lines_map: Dict[float, List[Dict]] = defaultdict(list)
    for w in words:
        y_key = round(w["top"] / 2.0) * 2.0
        lines_map[y_key].append(w)
    lines: List[Tuple[float, List[Dict]]] = [
        (y, sorted(ws, key=lambda w: w["x0"])) for y, ws in sorted(lines_map.items())
    ]
    # Determine tier column bounds
    tier_left, tier_right = compute_tier_bounds(words, page.width)
    # Identify y ranges that likely contain tier values to set header/footnote thresholds
    tier_line_ys = [
        y
        for y, line_words in lines
        if any(
            w["text"].strip().isdigit()
            and len(w["text"].strip()) == 1
            and tier_left <= w["x0"] <= tier_right
            for w in line_words
        )
    ]
    if tier_line_ys:
        first_tier_y = min(tier_line_ys)
        last_tier_y = max(tier_line_ys)
        top_threshold = first_tier_y - 20
        footnote_threshold = last_tier_y + 20
    else:
        top_threshold = footnote_threshold = None
    rows: List[Dict[str, str]] = []
    current_row = None
    family = current_family
    for y, line_words in lines:
        # Skip content outside the likely table region
        if top_threshold is not None and y < top_threshold:
            continue
        if footnote_threshold is not None and y > footnote_threshold:
            continue
        # Concatenate text for header/family detection
        texts = " ".join(w["text"] for w in line_words).strip()
        lower_text = texts.lower()
        # Skip table header lines
        if (
            ("drug" in lower_text and "name" in lower_text)
            or "tier" in lower_text
            or "requirements" in lower_text
            or "limits" in lower_text
        ):
            continue
        # Does the line contain any digit?
        has_digit = bool(re.search(r"\d", texts))
        # Collect tier tokens within the tier column bounds
        tier_tokens = [
            w["text"].strip()
            for w in line_words
            if w["text"].strip().isdigit()
            and len(w["text"].strip()) == 1
            and tier_left <= w["x0"] <= tier_right
        ]
        # Lines without digits
        if not has_digit:
            # Potential family heading: if the line is uppercase (letters and spaces) then
            # treat it as a new family even when a row is open.
            normalized = re.sub(r"\s+", " ", texts)
            # Check for potential category: uppercase, no digits, and no parentheses or semicolons
            # Determine if line should be treated as a category heading. Conditions:
            #   - Text is uppercase (ignoring multiple spaces)
            #   - Contains no digits or parentheses/semicolons
            #   - Does not contain substrings like ' OR ', ' AND ', or ' PF '
            #   - Has at least 3 non-space characters (to avoid single-letter tokens like 'LD')
            if (
                normalized
                and normalized == normalized.upper()
                and not re.search(r"[\d()\[\];]", normalized)
                and not any(kw in normalized for kw in [" OR ", " AND ", " PF "])
                and len(normalized.replace(" ", "")) >= 3
            ):
                # Close any ongoing row before switching families
                if current_row:
                    rows.append(current_row)
                    current_row = None
                # Remove surrounding parentheses from category names
                family = normalized.strip("()")
                continue
            # If no current row, treat as family heading even if not uppercase (e.g., first page)
            if current_row is None:
                # Remove surrounding parentheses when updating family outside row context
                family = normalized.strip("()")
                continue
            # Continuation line for current row (multi‑line name or requirements)
            name_tokens: List[str] = []
            req_tokens: List[str] = []
            for w in line_words:
                txt = w["text"].strip()
                if w["x1"] <= tier_left:
                    name_tokens.append(txt)
                elif w["x0"] >= tier_right:
                    req_tokens.append(txt)
                else:
                    # Ambiguous region: decide by punctuation/digit presence
                    if re.search(r"[;()\d]", txt):
                        req_tokens.append(txt)
                    else:
                        name_tokens.append(txt)
            if name_tokens:
                current_row["name"] += " " + " ".join(name_tokens)
            if req_tokens:
                if current_row["req"]:
                    current_row["req"] += " " + " ".join(req_tokens)
                else:
                    current_row["req"] = " ".join(req_tokens)
            continue
        # Lines with digits
        if tier_tokens:
            # Start a new row; close the previous row if any
            if current_row:
                rows.append(current_row)
                current_row = None
            tier_value = tier_tokens[0]
            name_tokens: List[str] = []
            req_tokens: List[str] = []
            for w in line_words:
                txt = w["text"].strip()
                # Skip the tier token itself
                if txt == tier_value and tier_left <= w["x0"] <= tier_right:
                    continue
                if w["x1"] <= tier_left:
                    name_tokens.append(txt)
                elif w["x0"] >= tier_right:
                    req_tokens.append(txt)
                else:
                    if re.search(r"[;()\d]", txt):
                        req_tokens.append(txt)
                    else:
                        name_tokens.append(txt)
            current_row = {
                "family": family,
                "name": " ".join(name_tokens).strip(),
                "tier": tier_value,
                "req": " ".join(req_tokens).strip(),
            }
            continue
        # Lines with digits but no tier token: continuation of current row
        if current_row is None:
            # Skip orphan lines (rare)
            continue
        name_tokens: List[str] = []
        req_tokens: List[str] = []
        for w in line_words:
            txt = w["text"].strip()
            if w["x1"] <= tier_left:
                name_tokens.append(txt)
            elif w["x0"] >= tier_right:
                req_tokens.append(txt)
            else:
                if re.search(r"[;()\d]", txt):
                    req_tokens.append(txt)
                else:
                    name_tokens.append(txt)
        if name_tokens:
            current_row["name"] += " " + " ".join(name_tokens)
        if req_tokens:
            if current_row["req"]:
                current_row["req"] += " " + " ".join(req_tokens)
            else:
                current_row["req"] = " ".join(req_tokens)
    # End of page: append any unfinished row
    if current_row:
        rows.append(current_row)
    return rows, family


def extract_aetna_details(input_pdf: str, output_excel: str) -> None:
    """Extract drug details from the Aetna formulary and write to Excel.

    Processes pages 8–73 (1‑indexed) and writes the results to an Excel
    workbook with columns: Drug Family, Drug Name, Drug Tier, Requirements/Limits.

    Args:
        input_pdf: Path to the Aetna PDF file.
        output_excel: Path to the output Excel file.
    """
    all_rows: List[Dict[str, str]] = []
    current_family: str = ""
    with pdfplumber.open(input_pdf) as pdf:
        # Pages 8–73 correspond to indices 7–72
        for page_index in range(7, 73):
            page_rows, current_family = parse_page(pdf.pages[page_index], current_family)
            all_rows.extend(page_rows)
    # Write the results to Excel
    workbook = xlsxwriter.Workbook(output_excel)
    worksheet = workbook.add_worksheet()
    headers = [
        "Drug Family",
        "Drug Name",
        "Drug Tier",
        "Requirements/Limits",
    ]
    for col, header in enumerate(headers):
        worksheet.write(0, col, header)
    for row_idx, row in enumerate(all_rows, start=1):
        worksheet.write(row_idx, 0, row.get("family", ""))
        worksheet.write(row_idx, 1, row.get("name", ""))
        worksheet.write(row_idx, 2, row.get("tier", ""))
        worksheet.write(row_idx, 3, row.get("req", ""))
    workbook.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract drug details from Aetna formulary PDF")
    parser.add_argument("--input", required=True, help="Path to the Aetna PDF file")
    parser.add_argument("--output", required=True, help="Path to the output Excel file")
    args = parser.parse_args()
    extract_aetna_details(args.input, args.output)


if __name__ == "__main__":
    main()