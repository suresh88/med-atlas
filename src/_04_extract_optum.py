"""
Extract drug information from the Optum formulary PDF using pdfplumber.

This module provides a command‑line script to parse a specific formulary
document distributed by Optum and produce a structured Excel file.  The PDF
contains two tables side‑by‑side on each page between pages 7 and 85 (1‑indexed).
Each table has three columns: ``Drug Name``, ``Drug Tier``, and ``Notes``.
Occasionally, a row will contain only a single value spanning all three
columns—these denote drug families.  For such rows, the family name should
propagate down to subsequent entries until the next family header appears.

The original extraction relied on the external ``pdftohtml`` utility from the
Poppler toolkit.  To avoid that dependency, this implementation uses
``pdfplumber`` to extract word‑level text and positional information directly
from the PDF.  It then reconstructs rows by clustering words by their vertical
positions and identifying the tier column based on the locations of digit
tokens.  A stateful parser builds complete rows, carrying forward pending
entries and drug families across pages.

Usage example:

    python extract_optum_drug_details_pdfplumber.py --input Optum.pdf --output optum_formulary.xlsx

Before running this script, ensure that ``pdfplumber`` and ``xlsxwriter`` are
installed in your Python environment.
"""

import argparse
import collections
import re
from typing import Dict, List, Tuple

import pdfplumber  # type: ignore
import xlsxwriter


def cluster_tier_positions(xs: List[int]) -> Tuple[float, float]:
    """Estimate the x coordinates of the ``Drug Tier`` columns by clustering digits.

    Given a list of x coordinates of text elements that are single digits, this
    function groups them into clusters based on proximity and frequency.  The
    two most populous clusters correspond to the left and right tables.  The
    centre positions of these clusters are returned in ascending order.

    Args:
        xs: A list of x coordinates for candidate digit elements.

    Returns:
        A tuple ``(left_center, right_center)`` representing the approximate x
        positions of the ``Drug Tier`` columns.  If fewer than two clusters are
        detected, both values will be equal.
    """
    if not xs:
        return 0.0, 0.0
    xs_sorted = sorted(xs)
    cluster_threshold = 50
    clusters: List[List[int]] = []
    current_cluster: List[int] = []
    for x in xs_sorted:
        if not current_cluster:
            current_cluster.append(x)
        else:
            if x - current_cluster[-1] <= cluster_threshold:
                current_cluster.append(x)
            else:
                clusters.append(current_cluster)
                current_cluster = [x]
    if current_cluster:
        clusters.append(current_cluster)
    cluster_info: List[Tuple[int, float]] = []
    for cluster in clusters:
        count = len(cluster)
        centre = sum(cluster) / count
        cluster_info.append((count, centre))
    cluster_info.sort(key=lambda c: c[0], reverse=True)
    if not cluster_info:
        return 0.0, 0.0
    if len(cluster_info) == 1:
        return cluster_info[0][1], cluster_info[0][1]
    centres = sorted([cluster_info[0][1], cluster_info[1][1]])
    return centres[0], centres[1]


def process_page_items_v2(
    items: List[Dict[str, int]],
    left_tier_x: float,
    right_tier_x: float,
    half_context: Dict[str, Dict[str, object]] = None,
) -> List[Dict[str, str]]:
    """Parse a page's text into structured drug entries using a stateful approach.

    This function reconstructs rows by accumulating text segments until both the
    drug name and tier have been encountered.  It maintains a pending entry for
    each table half (left and right) and finalises rows only when a tier digit
    is seen.  Lines that follow a completed row but contain no tier are appended
    to the last completed row's notes.  Category headings (drug families) are
    detected on the left side as a single text item without digits and reset
    the current family for both halves.

    Args:
        items: List of positioned text items with ``top``, ``left`` and ``text``.
        left_tier_x: Approximate x coordinate of the tier column in the left table.
        right_tier_x: Approximate x coordinate of the tier column in the right table.
        half_context: State dictionary carrying ``current_family``, ``pending`` and
            ``last_complete_idx`` for each half across pages.

    Returns:
        List of dictionaries with keys ``Drug Family``, ``Drug Name``, ``Drug Tier``
        and ``Notes`` for all completed rows on this page.
    """
    # Initialise context if necessary.  Each half holds:
    #  - current_family: active drug family
    #  - pending: a dict with keys family, name_parts, tier, notes_parts
    #  - last_complete_idx: index of last completed row in the results list
    if half_context is None:
        half_context = {
            'left': {'current_family': '', 'pending': None, 'last_complete_idx': None},
            'right': {'current_family': '', 'pending': None, 'last_complete_idx': None},
        }

    results: List[Dict[str, str]] = []

    # Helper to finalise a pending entry for a given half.
    def finalize_pending(half: str) -> None:
        hc = half_context[half]
        pending = hc.get('pending')
        if not pending:
            return
        # Only finalise if we have a tier value.
        if pending.get('tier') is None:
            return
        name = ' '.join(pending['name_parts']).strip()
        notes = ' '.join(pending['notes_parts']).strip()
        results.append({
            'Drug Family': pending['family'],
            'Drug Name': name,
            'Drug Tier': pending['tier'],
            'Notes': notes or None,
        })
        hc['last_complete_idx'] = len(results) - 1
        hc['pending'] = None

    # Cluster items by vertical positions into logical rows.  A larger threshold
    # (e.g., 25 pixels) is necessary because multi‑line names and tier values can
    # span lines separated by horizontal rules in the PDF.  If the threshold is
    # too small, the name, tier and notes may be split into separate rows and
    # erroneously classified as categories.
    row_threshold = 25  # pixels
    unique_tops = sorted({item['top'] for item in items})
    top_to_row: Dict[int, int] = {}
    cur_row_id = 0
    if unique_tops:
        cluster_start = unique_tops[0]
        top_to_row[cluster_start] = cur_row_id
        for top_y in unique_tops[1:]:
            if top_y - cluster_start <= row_threshold:
                top_to_row[top_y] = cur_row_id
            else:
                cur_row_id += 1
                top_to_row[top_y] = cur_row_id
                cluster_start = top_y
    rows: Dict[int, List[Tuple[int, str]]] = collections.defaultdict(list)
    for item in items:
        rows[top_to_row[item['top']]].append((item['left'], item['text']))

    # Compute split_x boundary to separate left and right tables.
    if right_tier_x > left_tier_x:
        split_x = left_tier_x + (right_tier_x - left_tier_x) * 0.4
    else:
        split_x = float('inf')

    # Iterate through each row.
    for row_id in sorted(rows.keys()):
        row_items = sorted(rows[row_id], key=lambda t: t[0])
        left_items: List[Tuple[int, str]] = []
        right_items: List[Tuple[int, str]] = []
        for x, txt in row_items:
            if x < split_x:
                left_items.append((x, txt))
            else:
                right_items.append((x, txt))
        # Determine if any digit token appears across the entire row.
        row_has_digit = any(re.fullmatch(r'\d', txt) for _, txt in row_items)
        # Build the full row text for header/category detection.
        row_text_lower = ' '.join(txt for _, txt in row_items).lower()
        # Skip header rows that contain column titles (e.g., "Drug Name", "Drug Tier", "Notes").
        if 'drug name' in row_text_lower or ('drug tier' in row_text_lower and ('note' in row_text_lower or 'notes' in row_text_lower)):
            continue
        # Detect category heading: if no digits appear anywhere on the row, then
        # treat the entire line as a drug family heading.  Category names span
        # multiple words and lines but contain no numeric tier values.  When
        # encountering such a row, finalise any pending entries and update the
        # current family for both halves.
        if not row_has_digit:
            finalize_pending('left')
            finalize_pending('right')
            family_name = ' '.join(txt for _, txt in row_items).strip()
            half_context['left']['current_family'] = family_name
            half_context['right']['current_family'] = family_name
            half_context['left']['last_complete_idx'] = None
            half_context['right']['last_complete_idx'] = None
            continue
        # Process each half individually.
        for half, items_in_half in [('left', left_items), ('right', right_items)]:
            if not items_in_half:
                continue
            joined = ' '.join(txt for _, txt in items_in_half)
            lower_joined = joined.lower()
            # Skip header lines containing column titles.
            if 'drug name' in lower_joined or ('tier' in lower_joined and ('note' in lower_joined or 'notes' in lower_joined)):
                continue
            hc = half_context[half]
            # Look for first single‑digit token in this half.
            tier_index = None
            for idx, (_, txt) in enumerate(items_in_half):
                if re.fullmatch(r'\d', txt):
                    tier_index = idx
                    break
            if tier_index is not None:
                name_parts = [txt for _, txt in items_in_half[:tier_index]]
                tier_value = items_in_half[tier_index][1]
                notes_parts = [txt for _, txt in items_in_half[tier_index + 1:]]
                if hc.get('pending') and hc['pending'].get('tier') is None:
                    hc['pending']['tier'] = tier_value
                    hc['pending']['notes_parts'].extend(notes_parts)
                    finalize_pending(half)
                else:
                    if hc.get('pending'):
                        finalize_pending(half)
                    if name_parts:
                        results.append({
                            'Drug Family': hc['current_family'],
                            'Drug Name': ' '.join(name_parts).strip(),
                            'Drug Tier': tier_value,
                            'Notes': ' '.join(notes_parts).strip() or None,
                        })
                        hc['last_complete_idx'] = len(results) - 1
                    else:
                        if hc.get('last_complete_idx') is not None:
                            text_to_append = ' '.join([tier_value] + notes_parts).strip()
                            if text_to_append:
                                prev_notes = results[hc['last_complete_idx']].get('Notes')
                                if prev_notes:
                                    results[hc['last_complete_idx']]['Notes'] = (prev_notes + ' ' + text_to_append).strip()
                                else:
                                    results[hc['last_complete_idx']]['Notes'] = text_to_append
                continue
            # No tier found in this half.
            if hc.get('pending'):
                hc['pending']['name_parts'].extend([txt for _, txt in items_in_half])
            else:
                if hc.get('last_complete_idx') is not None:
                    text_to_append = joined.strip()
                    if text_to_append:
                        prev_notes = results[hc['last_complete_idx']].get('Notes')
                        if prev_notes:
                            results[hc['last_complete_idx']]['Notes'] = (prev_notes + ' ' + text_to_append).strip()
                        else:
                            results[hc['last_complete_idx']]['Notes'] = text_to_append
                else:
                    hc['pending'] = {
                        'family': hc['current_family'],
                        'name_parts': [txt for _, txt in items_in_half],
                        'tier': None,
                        'notes_parts': [],
                    }
    return results


def extract_optum_formulary_pdfplumber(pdf_path: str, start_page: int, end_page: int) -> List[Dict[str, str]]:
    """Extract drug information from the specified page range of the Optum PDF using pdfplumber.

    This implementation uses pdfplumber's table extraction facilities to parse the
    two side‑by‑side tables on each page.  Each page is divided into left and
    right halves, and ``extract_tables`` is used with line‑detection to return
    cell contents.  Drug family headers are identified as rows with a single
    non‑empty cell, and the family name is propagated to subsequent rows until
    a new family header appears.  Data rows consist of three logical columns
    (name, tier, notes); if extra columns are detected, they are merged into
    the notes field after the tier value.

    Args:
        pdf_path: Path to the Optum PDF.
        start_page: First page number (1‑indexed) to process.
        end_page: Last page number (1‑indexed) to process, inclusive.

    Returns:
        A list of dictionaries representing the extracted rows with keys
        ``Drug Family``, ``Drug Name``, ``Drug Tier`` and ``Notes``.
    """
    results: List[Dict[str, str]] = []
    current_family = ''
    with pdfplumber.open(pdf_path) as pdf:
        for page_idx in range(start_page - 1, end_page):
            page = pdf.pages[page_idx]
            width = page.width
            height = page.height
            # Split the page into left and right halves.
            halves = [
                (0, 0, width / 2, height),
                (width / 2, 0, width, height),
            ]
            for (x0, y0, x1, y1) in halves:
                crop = page.crop((x0, y0, x1, y1))
                # Extract tables using line strategies.  pdfplumber will return
                # a list of tables, each table is a list of rows.  Each row is
                # a list of cell strings (or None) corresponding to columns.
                tables = crop.extract_tables({
                    'vertical_strategy': 'lines',
                    'horizontal_strategy': 'lines',
                })
                for table in tables:
                    for row in table:
                        if not row:
                            continue
                        # Strip whitespace and normalise None to empty strings.
                        cells = [(cell or '').strip() for cell in row]
                        # Remove trailing empty cells to simplify processing.
                        while cells and cells[-1] == '':
                            cells.pop()
                        if not cells:
                            continue
                        # Join row text for header detection.
                        row_text_lower = ' '.join(cells).lower()
                        # Skip header rows containing column titles.
                        if 'drug name' in row_text_lower and 'tier' in row_text_lower:
                            continue
                        # Identify category (drug family) rows: exactly one non‑empty cell.
                        non_empty_cells = [c for c in cells if c]
                        if len(non_empty_cells) == 1 and (cells[0] != ''):
                            # Update current family.
                            current_family = non_empty_cells[0]
                            continue
                        # Attempt to locate the tier column by finding the first
                        # single‑digit cell.  If not found, skip the row as it
                        # likely represents an unstructured line.
                        tier_index = None
                        for idx, cell in enumerate(cells):
                            # A valid tier is a single digit (e.g., 1‑5).
                            stripped = cell.replace(' ', '')
                            if stripped.isdigit() and len(stripped) == 1:
                                tier_index = idx
                                break
                        if tier_index is None:
                            # No tier found; skip row.
                            continue
                        # Build the name, tier and notes fields.  Name may span
                        # multiple cells before the tier column; notes may span
                        # one or more cells after the tier.
                        name = ' '.join(cells[:tier_index]).strip()
                        tier_value = cells[tier_index]
                        notes = ' '.join(cells[tier_index + 1:]).strip() if (tier_index + 1) < len(cells) else ''
                        results.append({
                            'Drug Family': current_family,
                            'Drug Name': name,
                            'Drug Tier': tier_value,
                            'Notes': notes or None,
                        })
    return results


def write_to_excel(rows: List[Dict[str, str]], excel_path: str) -> None:
    """Write extracted rows to an Excel file using xlsxwriter.

    Args:
        rows: A list of dictionaries representing the extracted rows.
        excel_path: Path to the Excel file to create.
    """
    workbook = xlsxwriter.Workbook(excel_path)
    worksheet = workbook.add_worksheet("Formulary")
    headers = ['Drug Family', 'Drug Name', 'Drug Tier', 'Notes']
    header_format = workbook.add_format({'bold': True})
    for col, header in enumerate(headers):
        worksheet.write(0, col, header, header_format)
    for row_idx, row in enumerate(rows, start=1):
        worksheet.write(row_idx, 0, row.get('Drug Family', ''))
        worksheet.write(row_idx, 1, row.get('Drug Name', ''))
        worksheet.write(row_idx, 2, row.get('Drug Tier', ''))
        worksheet.write(row_idx, 3, row.get('Notes', ''))
    workbook.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Optum formulary drug details from a PDF using pdfplumber.")
    parser.add_argument('--input', required=True, help='Path to the Optum PDF file')
    parser.add_argument('--output', required=True, help='Path to the output Excel file')
    parser.add_argument('--start_page', type=int, default=7, help='Start page number (1‑indexed)')
    parser.add_argument('--end_page', type=int, default=86, help='End page number (1‑indexed)')
    args = parser.parse_args()
    rows = extract_optum_formulary_pdfplumber(args.input, args.start_page, args.end_page)
    write_to_excel(rows, args.output)


if __name__ == '__main__':
    main()