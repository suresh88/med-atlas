"""
This script extracts drug details from an Anthem formulary PDF.

The PDF contains two tables per page across pages 9–60. Each table has three columns:
```
Drug Name | Drug Tier | Requirements/Limits
```
Drug family headings appear as un-tabulated rows with the three columns merged. This
script identifies those headings, assigns them to the `Drug Family` column and
propagates them to subsequent rows until a new heading is encountered. It also
handles cases where the right-hand table on the last pages may be missing.

Key features:
    * Uses pdfplumber to extract words and their coordinates, allowing precise
      reconstruction of multi-line rows and mixed-structured tables.
    * Detects the narrow “Tier” column by clustering single-digit words based on
      their x-coordinates. This excludes stray digits in headers and other text.
    * Merges continuation lines for both the `Drug Name` and `Requirements/Limits` fields.
    * Skips non-table text above the table header and below the final row of each table.
    * Writes the extracted data into an Excel file using xlsxwriter, avoiding
      heavy dependencies such as pandas.

Usage:
    python extract_drug_details.py --input Anthem.pdf --output anthem_formulary.xlsx

The output Excel file will contain columns: Drug Family, Drug Name, Drug Tier,
and Requirements/Limits with one row per drug entry.
"""

import argparse
import re
from collections import defaultdict
from typing import List, Tuple, Dict

import pdfplumber
import xlsxwriter


def compute_tier_bounds(words: List[Dict], crop_width: float) -> Tuple[float, float]:
    """Determine approximate x-coordinate boundaries for the Tier column.

    The Tier column contains single-digit numbers (1–5). To distinguish these
    digits from other numbers (e.g., phone numbers in headers), this function
    clusters x-coordinates of all single-digit words and selects the densest
    cluster. The minimum and maximum x0/x1 from that cluster become the
    boundaries of the Tier column.

    Args:
        words: List of word dictionaries from pdfplumber.extract_words().
        crop_width: Width of the cropped page region.

    Returns:
        A tuple (tier_left, tier_right) representing the left and right
        boundaries of the Tier column. Any word falling within these boundaries
        and consisting of a single digit is considered the tier value.
    """
    digit_words = [w for w in words if w["text"].strip().isdigit() and len(w["text"].strip()) == 1]
    if not digit_words:
        # fallback: assume Tier column is roughly mid-page if no digits found
        return crop_width * 0.4, crop_width * 0.5
    # Cluster x-coordinates by rounding to the nearest 10 units
    clusters: Dict[int, List[Dict]] = defaultdict(list)
    for w in digit_words:
        key = round(w["x0"] / 10) * 10
        clusters[key].append(w)
    # Select the cluster with the most entries (likely the Tier column)
    best_cluster = max(clusters.values(), key=len)
    tier_left = min(w["x0"] for w in best_cluster) - 2
    tier_right = max(w["x1"] for w in best_cluster) + 2
    return tier_left, tier_right


def parse_half(crop: pdfplumber.page.Page, initial_family: str = "") -> Tuple[List[Dict], str]:
    """Extract rows from a single half of a page.

    Args:
        crop: A pdfplumber Page object cropped to one half of the original page.
        initial_family: The drug family carried over from the previous half or page.

    Returns:
        A list of row dictionaries with fields: family, name, tier, req.
        The last detected family (to be passed to the next half or page).
    """
    words = crop.extract_words(x_tolerance=1, y_tolerance=1)
    if not words:
        return [], initial_family

    # Group words into lines by rounding their top coordinate
    lines_map: Dict[float, List[Dict]] = defaultdict(list)
    for w in words:
        y_key = round(w["top"] / 2.0) * 2.0
        lines_map[y_key].append(w)
    lines = [(y, sorted(ws, key=lambda w: w["x0"])) for y, ws in sorted(lines_map.items())]

    # Identify tier column boundaries and tier row positions
    tier_left, tier_right = compute_tier_bounds(words, crop.width)
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
        # Define thresholds to ignore header/footer text outside the table region
        top_threshold = first_tier_y - 20
        footnote_threshold = last_tier_y + 20
    else:
        first_tier_y = last_tier_y = None
        top_threshold = footnote_threshold = None

    current_family = initial_family
    current_row = None
    rows: List[Dict] = []
    candidate_lines: List[str] = []  # Potential family lines before data rows

    for y, line_words in lines:
        # Skip header and footer lines based on tier positions
        if first_tier_y is not None and y < top_threshold:
            continue
        if last_tier_y is not None and y > footnote_threshold:
            continue

        texts = " ".join(w["text"] for w in line_words).strip()
        lower_text = texts.lower()
        # Skip table header lines
        if (
            ("drug" in lower_text and ("name" in lower_text ))
            or "tier" in lower_text
            or "utilization" in lower_text
            or "management" in lower_text
            or "requirements" in lower_text
        ):
            continue

        has_digit = bool(re.search(r"\d", texts))
        tier_tokens = [
            w["text"].strip()
            for w in line_words
            if w["text"].strip().isdigit()
            and len(w["text"].strip()) == 1
            and tier_left <= w["x0"] <= tier_right
        ]

        # If no digits at all, determine whether this is a new drug family heading or a continuation line
        if not has_digit:
            # Normalize whitespace for family detection
            cleaned_text = re.sub(" +", " ", texts).strip()

            def is_potential_family(text: str) -> bool:
                """Heuristically decide if a line represents a drug family heading.

                A family heading typically contains only letters, spaces and hyphens, has no
                digits or special characters (e.g., semicolons, parentheses), and each word
                starts with an uppercase letter (title case). To reduce false positives on
                brand/generic names (which can also be title case), we additionally require
                the presence of certain keywords commonly found in category names such as
                "And", "Agents", "Modifiers", "Products", "Antineoplastics", or "Blood".
                Examples include:
                    "Analgesics And Anti-Inflammatory Agents"
                    "Antineoplastics"
                    "Blood Products And Modifiers"

                The check is case-sensitive: keywords must appear with an uppercase first
                letter (e.g., "And" rather than "and") which helps distinguish categories
                from generic names like "acetaminophen and codeine".
                """
                if not text:
                    return False
                # Reject if any disallowed characters present (digits or punctuation except hyphen)
                if re.search(r"[^A-Za-z\- ]", text):
                    return False
                words = text.split()
                # All words must start with uppercase letter
                if not (len(words) >= 1 and all(w[0].isupper() for w in words)):
                    return False
                # Keywords indicating a category/family
                # Do not treat generic combination drugs with "And" as categories.
                # Require at least one of these category-specific keywords.
                keywords = ["Agents", "Modifiers", "Products", "Antineoplastics", "Blood"]
                return any(kw in text for kw in keywords)

            # If this line looks like a family heading, update current_family
            if is_potential_family(cleaned_text):
                # Finalize any current row before switching families
                if current_row:
                    rows.append(current_row)
                    current_row = None
                current_family = cleaned_text
                # Clear any pending candidate lines
                candidate_lines = []
                continue

            # Not a family heading: treat as candidate line (before first data row) or continuation of current row
            if current_row is None:
                candidate_lines.append(cleaned_text)
                continue
            # Continuation line for a multi-line row
            name_tokens, req_tokens = [], []
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
            continue

        # If digits exist, check if this line introduces a tier (new data row)
        if tier_tokens:
            # Append the previous row before starting a new one
            if current_row:
                rows.append(current_row)
                current_row = None
            # If candidate family lines were collected, update current_family
            if candidate_lines:
                current_family = " ".join(candidate_lines).strip()
                candidate_lines = []
            tier_value = tier_tokens[0]
            name_tokens, req_tokens = [], []
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
                    # Ambiguous mid-region: decide based on content
                    if re.search(r"[;()\d]", txt):
                        req_tokens.append(txt)
                    else:
                        name_tokens.append(txt)
            current_row = {
                "family": current_family,
                "name": " ".join(name_tokens).strip(),
                "tier": tier_value,
                "req": " ".join(req_tokens).strip(),
            }
            continue

        # Continuation line with digits but not a new tier
        if current_row is None:
            continue
        name_tokens, req_tokens = [], []
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

    # Add any unfinished row
    if current_row:
        rows.append(current_row)
    return rows, current_family


def parse_page(page: pdfplumber.page.Page, initial_family: str) -> Tuple[List[Dict], str]:
    """Extract all rows from a full page by processing its two halves.

    Args:
        page: A pdfplumber Page object.
        initial_family: The drug family carried over from the previous page.

    Returns:
        A list of row dictionaries and the last detected family name for the next page.
    """
    width, height = page.width, page.height
    half_boxes = [(0, 0, width / 2, height), (width / 2, 0, width, height)]
    family = initial_family
    page_rows: List[Dict] = []
    for box in half_boxes:
        crop = page.crop(box)
        parsed_rows, family = parse_half(crop, family)
        page_rows.extend(parsed_rows)
    return page_rows, family


def extract_drug_details(input_pdf: str, output_excel: str) -> None:
    """Process the given PDF and write the extracted data to an Excel file.

    Args:
        input_pdf: Path to the input Anthem formulary PDF.
        output_excel: Path to the output Excel file.
    """
    all_rows: List[Dict] = []
    current_family = ""
    with pdfplumber.open(input_pdf) as pdf:
        # Pages 9–60 in the PDF (1-indexed) correspond to indices 8–59
        for page_index in range(8, 60):
            page_rows, current_family = parse_page(pdf.pages[page_index], current_family)
            all_rows.extend(page_rows)

    # Write the extracted data to an Excel file
    workbook = xlsxwriter.Workbook(output_excel)
    worksheet = workbook.add_worksheet()
    headers = ["Drug Family", "Drug Name", "Tier", "Requirements/Limits"]
    for col, header in enumerate(headers):
        worksheet.write(0, col, header)
    for row_idx, row in enumerate(all_rows, start=1):
        worksheet.write(row_idx, 0, row["family"])
        worksheet.write(row_idx, 1, row["name"])
        worksheet.write(row_idx, 2, row["tier"])
        worksheet.write(row_idx, 3, row["req"])
    workbook.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract drug details from an Anthem formulary PDF.")
    parser.add_argument("--input", required=True, help="Path to the Anthem formulary PDF")
    parser.add_argument("--output", required=True, help="Path to the output Excel file")
    args = parser.parse_args()
    extract_drug_details(args.input, args.output)


if __name__ == "__main__":
    main()