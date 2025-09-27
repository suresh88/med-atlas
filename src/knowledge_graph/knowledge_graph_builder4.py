"""
drug_graph_creator
===================

This script builds a knowledge graph for a single drug from a spreadsheet of
pharmaceutical data and exports the graph to a JSON file.  The graph captures
hierarchical relationships between ICD‑10 codes, diseases and drug metadata.

Usage example::

    python drug_graph_creator.py \
        --excel rxnav_with_fda_with_orig.xlsx \
        --drug BELBUCA \
        --json belbuca_graph.json

The script reads the specified Excel file, filters to the rows matching the
given drug name (case‑insensitive), constructs a directed graph of nodes and
edges capturing ICD‑10 hierarchies, disease mappings and drug information,
and writes the result as a JSON file.  Each node includes a ``type``
attribute and any additional metadata available in the source row.

The implementation uses a lightweight ``SimpleDiGraph`` class to store
nodes and edges.  If you have NetworkX installed you could adapt this code
to use it instead, but this self‑contained version avoids external
dependencies beyond pandas and the Python standard library.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from typing import Dict, List, Tuple

import pandas as pd


class SimpleDiGraph:
    """A minimal directed graph implementation.

    Nodes are stored in a dictionary mapping node identifiers to attribute
    dictionaries.  Edges are stored as a list of tuples (source, target,
    attribute_dict).  Basic methods are provided to add nodes and edges and
    query their existence.  This class is sufficient for constructing and
    exporting the knowledge graph without requiring NetworkX.
    """

    def __init__(self) -> None:
        self._nodes: Dict[str, Dict[str, any]] = {}
        self._edges: List[Tuple[str, str, Dict[str, any]]] = []

    def has_node(self, node: str) -> bool:
        return node in self._nodes

    def add_node(self, node: str, **attrs: any) -> None:
        if node not in self._nodes:
            self._nodes[node] = {}
        self._nodes[node].update(attrs)

    def nodes(self) -> Dict[str, Dict[str, any]]:
        return self._nodes

    def add_edge(self, u: str, v: str, **attrs: any) -> None:
        # Ensure nodes exist before adding edge
        if u not in self._nodes:
            self.add_node(u)
        if v not in self._nodes:
            self.add_node(v)
        self._edges.append((u, v, attrs.copy()))

    def edges(self) -> List[Tuple[str, str, Dict[str, any]]]:
        return self._edges


def parse_list_from_cell(cell: any) -> List[str]:
    """Attempt to parse a list of strings from a cell.

    The cell may contain a Python list representation or a string of items
    separated by semicolons or commas.  If parsing fails or the cell is
    missing, an empty list is returned.
    """
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return []
    if isinstance(cell, list):
        return [str(item).strip() for item in cell]
    if isinstance(cell, str):
        try:
            value = ast.literal_eval(cell)
            if isinstance(value, list):
                return [str(item).strip() for item in value]
        except Exception:
            pass
        # fall back to splitting on common delimiters
        return [item.strip() for item in re.split(r'[;,]', cell) if item.strip()]
    return []


def build_drug_graph(excel_path: str, drug_name: str) -> SimpleDiGraph:
    """Build a knowledge graph for a specific drug from an Excel file.

    Parameters
    ----------
    excel_path : str
        Path to the Excel spreadsheet containing the source data.
    drug_name : str
        Name of the drug to filter on (case‑insensitive).

    Returns
    -------
    SimpleDiGraph
        A directed graph representing the drug's relationships to diseases,
        ICD‑10 codes, ingredients, dose forms and metadata.
    """
    df = pd.read_excel(excel_path)
    # case‑insensitive match on Drug Name column
    mask = df['Drug Name'].fillna('').astype(str).str.strip().str.upper() == drug_name.strip().upper()
    df_drug = df.loc[mask]
    if df_drug.empty:
        raise ValueError(f"Drug '{drug_name}' not found in file {excel_path}")

    G = SimpleDiGraph()

    for _, row in df_drug.iterrows():
        drug = row['Drug Name'].strip()
        # Add the drug node with available metadata
        drug_attrs = {
            'type': 'Drug',
            'formulary': row.get('Formulary'),
            'rxcui': row.get('RxCUI'),
            'brand_name': row.get('Brand Name'),
            'branded_dose_form_group': row.get('Branded Dose Form Group'),
            'branded_drug_component': row.get('Branded Drug Component'),
            'branded_drug_or_pack': row.get('Branded Drug or Pack'),
            'clinical_dose_form_group': row.get('Clinical Dose Form Group'),
            'clinical_drug_component': row.get('Clinical Drug Component'),
            'clinical_drug_or_pack': row.get('Clinical Drug or Pack'),
            'dose_form_group': row.get('Dose Form Group'),
            'ingredient': row.get('Ingredient'),
            'precise_ingredient': row.get('Precise Ingredient'),
            'orig_approval_date': row.get('orig_approval_date'),
        }
        G.add_node(drug, **drug_attrs)

        # Parse the ICD mapping for may_treat_diseases_icd10
        icd_mapping = row.get('may_treat_diseases_icd10')
        icd_map = []
        if pd.notna(icd_mapping):
            try:
                icd_map = ast.literal_eval(icd_mapping)
            except Exception:
                icd_map = []

        # Build disease and ICD hierarchy
        for mapping in icd_map:
            if not isinstance(mapping, dict):
                continue
            for disease, codes in mapping.items():
                disease_name = disease.strip()
                G.add_node(disease_name, type='Disease')
                G.add_edge(drug, disease_name, relation='TREATS')
                for code in codes:
                    code_clean = str(code).strip().upper()
                    if not code_clean:
                        continue
                    # Root is first three characters before dot
                    root = code_clean.split('.')[0]
                    root = root[:3]
                    G.add_node(root, type='ICD_Root')
                    G.add_node(code_clean, type='ICD_Code')
                    G.add_edge(root, code_clean, relation='BELONGS_TO')
                    G.add_edge(code_clean, disease_name, relation='DIAGNOSED_AS')

        # Ingredients and precise ingredients
        for ing in parse_list_from_cell(row.get('Ingredient')):
            G.add_node(ing, type='Ingredient')
            G.add_edge(drug, ing, relation='CONTAINS')
        for ping in parse_list_from_cell(row.get('Precise Ingredient')):
            G.add_node(ping, type='Precise_Ingredient')
            G.add_edge(drug, ping, relation='CONTAINS')

        # Brand name: connect if different from drug name
        brand = row.get('Brand Name')
        if pd.notna(brand) and brand.strip() and brand.strip() != drug:
            brand_name = brand.strip()
            G.add_node(brand_name, type='Brand')
            G.add_edge(drug, brand_name, relation='BRANDED_AS')

        # Formulary
        formulary = row.get('Formulary')
        if pd.notna(formulary) and str(formulary).strip():
            form_name = str(formulary).strip()
            G.add_node(form_name, type='Formulary')
            G.add_edge(drug, form_name, relation='FORMULATED_AS')

        # Dose forms and related fields
        dose_fields = [
            'Branded Dose Form Group',
            'Branded Drug Component',
            'Branded Drug or Pack',
            'Clinical Dose Form Group',
            'Clinical Drug Component',
            'Clinical Drug or Pack',
            'Dose Form Group',
        ]
        for col in dose_fields:
            val = row.get(col)
            if pd.notna(val) and str(val).strip():
                node_id = str(val).strip()
                G.add_node(node_id, type='Dose_Form')
                G.add_edge(drug, node_id, relation='FORMULATED_AS')

        # Approval date node
        approval = row.get('orig_approval_date')
        if pd.notna(approval):
            date_str = str(approval)
            node_id = f'Approval Date: {date_str}'
            G.add_node(node_id, type='Approval_Date', approval_date=date_str)
            G.add_edge(drug, node_id, relation='APPROVED_ON')

    return G


def export_graph_to_json(G: SimpleDiGraph, filepath: str) -> None:
    """Write the graph to a JSON file.

    Parameters
    ----------
    G : SimpleDiGraph
        The graph to export.
    filepath : str
        Path to the output JSON file.
    """
    data = {
        'nodes': [
            {'id': node, **attrs}
            for node, attrs in G.nodes().items()
        ],
        'edges': [
            {'source': u, 'target': v, **attrs}
            for u, v, attrs in G.edges()
        ],
    }
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


def plot_drug_graph(G: SimpleDiGraph, title: str | None = None) -> None:
    """Visualize the knowledge graph for a single drug using a radial layout.

    This helper arranges nodes by type into concentric circles: the drug at
    the centre, its metadata (approval date, brand, formulary, dose forms,
    ingredients) on the next ring, diseases on the third ring, ICD codes on
    the fourth ring and ICD root categories on the outer ring.  Only basic
    node labels (drug and diseases) are displayed to avoid clutter.

    Parameters
    ----------
    G : SimpleDiGraph
        The graph to visualize.
    title : str, optional
        Title for the plot.

    Notes
    -----
    Requires ``matplotlib`` to be installed.  The function will raise
    ImportError if matplotlib is not available.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError('matplotlib is required for plotting')
    import math

    # Assign levels based on node type
    levels: Dict[str, int] = {}
    for node, attrs in G.nodes().items():
        t = attrs.get('type')
        if t == 'Drug':
            lvl = 0
        elif t in ('Approval_Date', 'Brand', 'Formulary', 'Dose_Form', 'Ingredient', 'Precise_Ingredient'):
            lvl = 1
        elif t == 'Disease':
            lvl = 2
        elif t == 'ICD_Code':
            lvl = 3
        elif t == 'ICD_Root':
            lvl = 4
        else:
            lvl = 5
        levels[node] = lvl

    # Organize nodes by level
    nodes_by_level: Dict[int, List[str]] = {}
    for node, lvl in levels.items():
        nodes_by_level.setdefault(lvl, []).append(node)

    # Compute radial positions
    pos: Dict[str, Tuple[float, float]] = {}
    for lvl, nodes_list in nodes_by_level.items():
        n = len(nodes_list)
        if n == 0:
            continue
        # Distance of each level from the centre
        radius = 1 + lvl * 1.5
        for i, node in enumerate(nodes_list):
            angle = 2 * math.pi * i / n
            pos[node] = (radius * math.cos(angle), radius * math.sin(angle))

    # Plot using matplotlib
    plt.figure(figsize=(8, 8))
    # Edge drawing
    for u, v, attrs in G.edges():
        x0, y0 = pos.get(u, (0, 0))
        x1, y1 = pos.get(v, (0, 0))
        plt.plot([x0, x1], [y0, y1], color='gray', linewidth=0.5)
    # Node drawing with color by type
    color_map = {
        'Drug': 'blue',
        'Approval_Date': 'cyan',
        'Brand': 'orange',
        'Formulary': 'brown',
        'Dose_Form': 'lightgray',
        'Ingredient': 'green',
        'Precise_Ingredient': 'lime',
        'Disease': 'red',
        'ICD_Code': 'purple',
        'ICD_Root': 'pink',
    }
    for node, (x, y) in pos.items():
        t = G.nodes()[node].get('type')
        plt.scatter([x], [y], color=color_map.get(t, 'black'), s=50)
    # Label the drug and disease nodes only
    for node, (x, y) in pos.items():
        t = G.nodes()[node].get('type')
        if t in ('Drug', 'Disease'):
            plt.text(x, y, node, fontsize=8, ha='center', va='center')
    plt.axis('off')
    if title:
        plt.title(title)
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description='Build a knowledge graph for a specific drug and export to JSON.')
    parser.add_argument('--excel', required=True, help='Path to the Excel file.')
    parser.add_argument('--drug', required=True, help='Drug name to build the graph for.')
    parser.add_argument('--json', required=True, help='Output JSON file path.')
    args = parser.parse_args()

    G = build_drug_graph(args.excel, args.drug)
    export_graph_to_json(G, args.json)
    print(f"Graph built for '{args.drug}' and exported to {args.json}")


if __name__ == '__main__':
    main()