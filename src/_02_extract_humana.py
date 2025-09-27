"""
Extract drug details from the Humana formulary PDF.

The Humana formulary lists medications in a tabular format across pages 11–84
(1‑indexed). Each page contains a single table with three logical columns:

    • **DRUG NAME** – The medication and formulation details.
    • **TIER** – A tier number (1–5) or code (e.g. DL, MO) indicating coverage.
    • **UTILIZATION MANAGEMENT REQUIREMENTS** – Notes such as PA, QL and limits.

Drug family or category headings (e.g. “ANALGESICS”) appear as standalone
rows where only the first column is filled and the tier and requirements
columns are blank. These headings apply to the rows that follow until
another heading is encountered.

This script uses pdfplumber's high‑level ``extract_tables`` function with
``vertical_strategy='lines'`` and ``horizontal_strategy='lines'`` to convert
each page into a list of rows. The function correctly spans multi‑line
cells and recognises the three columns. For each row, the script determines
whether it represents a drug entry or a family heading and builds a list of
records accordingly. The final output is written to an Excel file.

Usage example:

    python extract_humana_drug_details.py --input Humana.pdf --output humana_drug_details.xlsx

Dependencies: pdfplumber, xlsxwriter
"""

import argparse
import re
from typing import List, Dict

import pdfplumber
import xlsxwriter


def extract_humana_details(input_pdf: str, output_excel: str) -> None:
    """Read a Humana formulary PDF and export drug details to Excel.

    The function processes pages 11–84 (1‑indexed) of the input PDF. It
    interprets rows with an empty tier and requirements as drug family
    headings and propagates that family name to subsequent rows until a new
    heading is encountered.

    Args:
        input_pdf: Path to the Humana PDF file.
        output_excel: Destination path for the generated Excel file.
    """
    rows: List[Dict[str, str]] = []
    current_family: str = ""
    with pdfplumber.open(input_pdf) as pdf:
        # Pages 11–84 correspond to indices 10–83 in zero‑based indexing
        for page_index in range(10, 84):
            page = pdf.pages[page_index]
            # Extract table(s) using line‑based strategies; expect one table per page
            tables = page.extract_tables(
                {"vertical_strategy": "lines", "horizontal_strategy": "lines"}
            )
            if not tables:
                continue
            table = tables[0]
            for row in table:
                # Skip completely empty rows
                if not row or all(cell is None or str(cell).strip() == "" for cell in row):
                    continue
                # Normalise cells to strings (None becomes empty string)
                first = (row[0] or "").strip()
                tier = (row[1] or "").strip()
                req = (row[2] or "").strip() if len(row) > 2 else ""
                # Lowercase version of first cell for header detection
                lower_first = first.lower()
                # Skip the header row containing the column names
                if ("drug" in lower_first and "name" in lower_first) or (
                    "tier" in lower_first and "utilization" in lower_first
                ):
                    continue
                # Detect family heading: non‑empty first column with blank tier and req
                if first and not tier and not req:
                    # Collapse multiple spaces
                    current_family = re.sub(r"\s+", " ", first)
                    continue
                # Otherwise, it's a data row. Use the current family.
                # Collapse internal whitespace in drug name and requirements
                cleaned_name = re.sub(r"\s+", " ", first)
                cleaned_req = re.sub(r"\s+", " ", req)
                rows.append(
                    {
                        "family": current_family,
                        "name": cleaned_name,
                        "tier": tier,
                        "req": cleaned_req,
                    }
                )
    # Write the results to an Excel file
    workbook = xlsxwriter.Workbook(output_excel)
    worksheet = workbook.add_worksheet()
    headers = [
        "Drug Family",
        "Drug Name",
        "Tier",
        "Utilization Management Requirements",
    ]
    for col, header in enumerate(headers):
        worksheet.write(0, col, header)
    for row_idx, record in enumerate(rows, start=1):
        worksheet.write(row_idx, 0, record["family"])
        worksheet.write(row_idx, 1, record["name"])
        worksheet.write(row_idx, 2, record["tier"])
        worksheet.write(row_idx, 3, record["req"])
    workbook.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract drug details from pages 11–84 of the Humana formulary PDF."
    )
    parser.add_argument("--input", required=True, help="Path to the input PDF file")
    parser.add_argument("--output", required=True, help="Path to the output Excel file")
    args = parser.parse_args()
    extract_humana_details(args.input, args.output)


if __name__ == "__main__":
    main()