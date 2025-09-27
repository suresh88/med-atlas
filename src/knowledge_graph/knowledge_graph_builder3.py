"""
knowledge_graph_builder
========================

This module provides functionality to construct a comprehensive knowledge
graph from a spreadsheet containing drug information, disease relationships,
ICD‑10 codes and FDA approval metadata.  It uses the pandas library for
data ingestion and cleansing, the built‑in `ast` module to parse string
representations of Python objects, and NetworkX to build the directed
knowledge graph.  In addition to graph construction, the module performs
basic data quality checks, produces summary statistics and optionally
exports the graph to various formats (GraphML, GEXF, JSON).  A simple
matplotlib/plotly visualization is also provided for exploring a subset of
the network.

The high‑level workflow executed by the ``build_knowledge_graph`` function
mirrors the steps described in the user task:

1. Load and inspect the Excel file using pandas.
2. Clean and standardise the text fields and parse complex columns such
   as the ICD‑10 disease mappings and FDA approval metadata.
3. Build a hierarchical representation of ICD‑10 codes: root (first
   three characters), full code and disease names.
4. Connect each drug to relevant disease nodes using the relationship
   types MAY_PREVENT, CONTRAINDICATED_WITH, MAY_DIAGNOSE and MAY_TREAT.
5. Integrate FDA approval information by creating nodes for approved
   submissions and linking them back to the corresponding drug.
6. Perform data quality checks (invalid codes, orphan nodes, etc.) and
   summarise the results.
7. Return the resulting NetworkX graph alongside summary statistics and
   optionally save the graph to disk.

Example usage::

    from knowledge_graph_builder import build_knowledge_graph
    G, stats, report = build_knowledge_graph('/home/oai/share/rxnav_with_fda.xlsx')
    # view basic statistics
    print(stats)
    # export to graphml
    nx.write_graphml(G, 'drug_graph.graphml')

Note:
-----
This module intentionally avoids imposing an external schema on the
graph.  Instead, node and edge attributes are stored as dictionaries
attached to each entity.  Consumers of the resulting graph can query
these attributes as needed.  When exporting to GraphML or GEXF, these
attributes are preserved.
"""

from __future__ import annotations

import ast
import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
import math
import matplotlib.pyplot as plt  # used for fallback static plots
import plotly.graph_objects as go  # used for interactive visualisations

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


ICD10_PATTERN = re.compile(r"^[A-TV-Z][0-9]{2}(?:\.[0-9A-TV-Z]{1,4})?$")


class SimpleDiGraph:
    """
    A very lightweight directed graph implementation intended as a drop‑in
    replacement for the subset of NetworkX functionality needed in this
    project.  It supports node and edge addition, attribute storage and
    basic queries such as predecessors, successors, degree counts and
    subgraph extraction.  Graph exports are handled via helper functions
    rather than methods on this class.
    """

    def __init__(self) -> None:
        # Node storage: id -> attribute dict
        self._nodes: Dict[str, Dict[str, any]] = {}
        # Edge storage: list of (source, target, attrs)
        self._edges: List[Tuple[str, str, Dict[str, any]]] = []
        # Adjacency: source -> set of targets
        self._succ: Dict[str, set[str]] = defaultdict(set)
        # Reverse adjacency: target -> set of sources
        self._pred: Dict[str, set[str]] = defaultdict(set)

    # Node methods
    def has_node(self, node: str) -> bool:
        return node in self._nodes

    def add_node(self, node: str, **attrs: any) -> None:
        if node not in self._nodes:
            self._nodes[node] = {}
        # update attributes
        self._nodes[node].update(attrs)

    def nodes(self) -> Dict[str, Dict[str, any]]:
        return self._nodes

    def number_of_nodes(self) -> int:
        return len(self._nodes)

    # Edge methods
    def has_edge(self, u: str, v: str) -> bool:
        return v in self._succ.get(u, set())

    def add_edge(self, u: str, v: str, **attrs: any) -> None:
        # ensure nodes exist
        if u not in self._nodes:
            self.add_node(u)
        if v not in self._nodes:
            self.add_node(v)
        # store edge
        if not self.has_edge(u, v):
            self._edges.append((u, v, attrs.copy()))
            self._succ[u].add(v)
            self._pred[v].add(u)
        else:
            # If the edge already exists, update its attributes
            for i, (src, dst, edata) in enumerate(self._edges):
                if src == u and dst == v:
                    edata.update(attrs)
                    self._edges[i] = (src, dst, edata)
                    break

    def edges(self) -> List[Tuple[str, str, Dict[str, any]]]:
        return self._edges

    def number_of_edges(self) -> int:
        return len(self._edges)

    def get_edge_data(self, u: str, v: str) -> Optional[Dict[str, any]]:
        """Return the attribute dictionary for the edge (u, v), or None if
        the edge does not exist."""
        for src, dst, attrs in self._edges:
            if src == u and dst == v:
                return attrs
        return None

    # Graph queries
    def successors(self, node: str) -> List[str]:
        return list(self._succ.get(node, set()))

    def predecessors(self, node: str) -> List[str]:
        return list(self._pred.get(node, set()))

    def degree(self, node: str) -> int:
        return len(self._succ.get(node, set())) + len(self._pred.get(node, set()))

    def subgraph(self, nodes: set[str]):
        # Create a new SimpleDiGraph containing only nodes and edges among them
        subG = SimpleDiGraph()
        for node in nodes:
            if node in self._nodes:
                subG.add_node(node, **self._nodes[node])
        for u, v, attrs in self._edges:
            if u in nodes and v in nodes:
                subG.add_edge(u, v, **attrs)
        return subG


@dataclass
class DataQualityReport:
    """Data structure to hold summary statistics and quality issues."""

    total_rows: int = 0
    total_drug_nodes: int = 0
    total_disease_nodes: int = 0
    total_icd_root_nodes: int = 0
    total_icd_code_nodes: int = 0
    total_fda_nodes: int = 0
    total_edges: int = 0
    invalid_icd_codes: List[str] = field(default_factory=list)
    orphan_disease_nodes: List[str] = field(default_factory=list)
    orphan_drug_nodes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, any]:
        return {
            'total_rows': self.total_rows,
            'total_drug_nodes': self.total_drug_nodes,
            'total_disease_nodes': self.total_disease_nodes,
            'total_icd_root_nodes': self.total_icd_root_nodes,
            'total_icd_code_nodes': self.total_icd_code_nodes,
            'total_fda_nodes': self.total_fda_nodes,
            'total_edges': self.total_edges,
            'invalid_icd_codes': self.invalid_icd_codes,
            'orphan_disease_nodes': self.orphan_disease_nodes,
            'orphan_drug_nodes': self.orphan_drug_nodes,
        }


def parse_icd_mapping(cell: Optional[str]) -> Dict[str, List[str]]:
    """Safely parse a cell containing a string representation of a list of
    dictionaries mapping disease names to lists of ICD‑10 codes.

    Parameters
    ----------
    cell : str or None
        The cell value from the DataFrame.

    Returns
    -------
    mapping : Dict[str, List[str]]
        A dictionary mapping disease names to lists of ICD codes.
    """
    if cell is None or (isinstance(cell, float) and pd.isna(cell)) or cell == '':
        return {}
    try:
        # Use ast.literal_eval because the string uses single quotes and may
        # represent Python literals rather than valid JSON.  literal_eval will
        # safely evaluate lists/dicts composed of basic types.
        data = ast.literal_eval(cell)
        mapping: Dict[str, List[str]] = {}
        for entry in data:
            if isinstance(entry, dict):
                for disease, codes in entry.items():
                    if isinstance(codes, list):
                        mapping.setdefault(disease.strip(), []).extend([c.strip() for c in codes])
        return mapping
    except Exception as exc:
        logger.warning(f"Failed to parse ICD mapping from cell: {exc}")
        return {}


def extract_icd_hierarchy(icd_mappings: List[Dict[str, List[str]]]) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    """Extract unique ICD‑10 codes and their hierarchical root prefixes.

    Parameters
    ----------
    icd_mappings : list of dict
        A list where each element is a mapping of disease names to lists of ICD codes.

    Returns
    -------
    code_to_root : dict
        Mapping from full ICD code to its three character root (category).
    disease_to_codes : dict
        Mapping from disease name to list of ICD codes associated with that disease.
    """
    code_to_root: Dict[str, str] = {}
    disease_to_codes: Dict[str, List[str]] = defaultdict(list)
    for mapping in icd_mappings:
        for disease, codes in mapping.items():
            for code in codes:
                # Validate code format and extract root
                code_clean = code.strip().upper()
                if code_clean:
                    root = code_clean[:3]  # first three characters as category
                    code_to_root[code_clean] = root
                    disease_to_codes[disease].append(code_clean)
    return code_to_root, disease_to_codes


def parse_fda_data(cell: Optional[str]) -> List[Dict[str, any]]:
    """Parse the FDA approval column into a list of dictionaries.

    Parameters
    ----------
    cell : str or None
        The string representation of a JSON list.

    Returns
    -------
    records : list of dict
        A list of dictionaries extracted from the JSON string.  If the
        string cannot be parsed, an empty list is returned.
    """
    if cell is None or (isinstance(cell, float) and pd.isna(cell)) or cell == '':
        return []
    try:
        # The FDA column is valid JSON; use json.loads
        return json.loads(cell)
    except Exception as exc:
        logger.warning(f"Failed to parse FDA data: {exc}")
        return []


def build_knowledge_graph(excel_path: str, drug_name: Optional[str] = None):
    """Main entry point for constructing the drug–disease–ICD–FDA knowledge graph.

    This function orchestrates reading the Excel data, parsing complex
    structures, building hierarchical and relational graph components and
    computing summary statistics.

    Parameters
    ----------
    excel_path : str
        The path to the Excel file containing the source data.
    drug_name : Optional[str], optional
        If provided, only build the graph for the specified drug.  The match
        is case‑insensitive and trims whitespace.  When ``None`` (default),
        the graph is built for all drugs in the file.

    Returns
    -------
    graph : nx.DiGraph
        The constructed directed graph with nodes and edges.
    stats : dict
        A dictionary containing high‑level statistics about the graph.
    report : DataQualityReport
        An object summarising data quality issues and counts.
    """
    # Load data
    df = pd.read_excel(excel_path)
    # If a specific drug name is requested, filter the DataFrame accordingly.
    # The comparison is case‑insensitive and strips whitespace to avoid
    # mismatches due to formatting differences.
    if drug_name is not None:
        # Ensure drug_name is a stripped lowercase string
        target = str(drug_name).strip().lower()
        # Create a mask where the 'Drug Name' column matches the target
        mask = df['Drug Name'].fillna('').astype(str).str.strip().str.lower() == target
        df = df.loc[mask].copy()
        logger.info(f"Filtered to {len(df)} row(s) for drug '{drug_name}' from {excel_path}")
    else:
        logger.info(f"Loaded {len(df)} rows from {excel_path}")

    # Initialise graph using a simple custom implementation rather than
    # NetworkX (which may not be available in the execution environment).
    G = SimpleDiGraph()

    # Keep track of nodes to avoid duplicates
    icd_root_nodes: set[str] = set()
    icd_code_nodes: set[str] = set()
    disease_nodes: set[str] = set()
    drug_nodes: set[str] = set()
    fda_nodes: set[str] = set()

    report = DataQualityReport(total_rows=len(df))

    # Pre‑parse ICD mappings.  We restrict to only one relationship:
    # MAY_TREAT, as per the updated requirements.  Other relationship types
    # (MAY_PREVENT, CONTRAINDICATED_WITH, MAY_DIAGNOSE) are ignored to keep
    # the graph consumable.  See instructions for details.
    relation_cols = {
        'MAY_TREAT': 'may_treat_diseases_icd10',
    }

    # No precomputation of ICD mappings is necessary; mappings will be parsed
    # on a per‑row basis inside the main iteration.  See the row loop below.

    # Iterate through rows to build nodes and edges
    for idx, row in df.iterrows():
        # Build drug node
        drug_name = str(row['Drug Name']).strip()
        if not drug_name:
            continue
        drug_nodes.add(drug_name)
        if not G.has_node(drug_name):
            # Collect drug attributes (include many relevant columns)
            drug_attrs = {
                'type': 'Drug',
                'drug_family': row.get('Drug Family'),
                'drug_tier': int(row['Drug Tier']) if not pd.isna(row['Drug Tier']) else None,
                'notes': row.get('Notes'),
                'formulary': row.get('Formulary'),
                'match_variant': row.get('Match Variant'),
                'rx_cui': int(row['RxCUI']) if not pd.isna(row['RxCUI']) else None,
                'matched_name': row.get('Matched Name'),
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
                'search_term': row.get('Search Term'),
                'last_updated': row.get('last_updated'),
            }
            G.add_node(drug_name, **drug_attrs)

        # For each relation type, parse disease→codes mapping.  We compute
        # the mapping on the fly from the current row rather than using a
        # precomputed structure.  This avoids issues when filtering to a
        # single drug.
        for rel, col in relation_cols.items():
            cell_val = row.get(col)
            mapping = parse_icd_mapping(cell_val)
            for disease, codes in mapping.items():
                disease_clean = disease.strip()
                if not disease_clean:
                    continue
                # Add disease node
                if disease_clean not in disease_nodes:
                    disease_nodes.add(disease_clean)
                    G.add_node(disease_clean, type='Disease', name=disease_clean)

                # For each code, add code and root nodes
                for code in codes:
                    code_clean = code.strip().upper()
                    if not code_clean:
                        continue
                    root = code_clean[:3]
                    # Validate format
                    if not ICD10_PATTERN.match(code_clean):
                        report.invalid_icd_codes.append(code_clean)
                    # Add root node
                    if root not in icd_root_nodes:
                        icd_root_nodes.add(root)
                        G.add_node(root, type='ICD_Root', name=root)
                    # Add full code node
                    if code_clean not in icd_code_nodes:
                        icd_code_nodes.add(code_clean)
                        G.add_node(code_clean, type='ICD_Code', name=code_clean, root=root)
                        # Connect root to code
                        G.add_edge(root, code_clean, relation='HAS_SUB_CODE')
                    # Connect code to disease
                    if not G.has_edge(code_clean, disease_clean):
                        G.add_edge(code_clean, disease_clean, relation='MAPS_TO_DISEASE')
                    # Connect drug to disease with specific relation
                    # We'll orient edge from drug to disease; store relation as attribute
                    G.add_edge(drug_name, disease_clean, relation=rel)

        # Parse FDA approval (original submission) information and add nodes/edges.
        # We use the extracted orig_* columns rather than the full fda_approval_data
        # column to avoid multiple submissions and keep the graph simple.
        app_no = row.get('orig_application_number')
        approval_date = row.get('orig_approval_date')
        # Only create a node when both application number and approval date are present
        if pd.notna(app_no) and pd.notna(approval_date):
            app_no_str = str(app_no).strip()
            approval_date_str = str(approval_date)
            node_id = f"FDA:{app_no_str}:{approval_date_str}"
            if node_id not in fda_nodes:
                fda_nodes.add(node_id)
                # Attempt to parse the approval date into a datetime object
                date_obj = None
                try:
                    date_obj = pd.to_datetime(approval_date, errors='coerce')
                    if pd.isna(date_obj):
                        date_obj = None
                except Exception:
                    date_obj = None
                # Collect additional metadata from orig_* columns if available
                orig_type = row.get('orig_submission_type')
                orig_status = row.get('orig_submission_status')
                orig_number = row.get('orig_submission_number')
                orig_brand_names = row.get('orig_brand_names')
                orig_generic_names = row.get('orig_generic_names')
                G.add_node(
                    node_id,
                    type='FDA_Approval',
                    application_number=app_no_str,
                    approval_date=approval_date_str,
                    submission_type=orig_type,
                    submission_status=orig_status,
                    submission_number=orig_number,
                    brand_names=orig_brand_names,
                    generic_names=orig_generic_names,
                    date=date_obj
                )
            # Connect drug to FDA approval
            G.add_edge(drug_name, node_id, relation='HAS_FDA_APPROVAL')

    # Data quality: identify orphan nodes (diseases not connected to codes)
    for disease in disease_nodes:
        # A disease is considered orphan if it has no incoming ICD mapping edges
        has_mapping = False
        for u in G.predecessors(disease):
            edata = G.get_edge_data(u, disease)
            if edata and edata.get('relation') == 'MAPS_TO_DISEASE':
                has_mapping = True
                break
        if not has_mapping:
            report.orphan_disease_nodes.append(disease)
    # identify orphan drugs (drugs with no relations)
    for drug in drug_nodes:
        if G.degree(drug) == 0:
            report.orphan_drug_nodes.append(drug)

    # Populate report counts
    report.total_drug_nodes = len(drug_nodes)
    report.total_disease_nodes = len(disease_nodes)
    report.total_icd_root_nodes = len(icd_root_nodes)
    report.total_icd_code_nodes = len(icd_code_nodes)
    report.total_fda_nodes = len(fda_nodes)
    report.total_edges = G.number_of_edges()

    # Prepare high level stats for convenience
    stats = {
        'num_nodes': G.number_of_nodes(),
        'num_edges': G.number_of_edges(),
        'num_drugs': len(drug_nodes),
        'num_diseases': len(disease_nodes),
        'num_icd_codes': len(icd_code_nodes),
        'num_icd_roots': len(icd_root_nodes),
        'num_fda_approvals': len(fda_nodes),
    }
    return G, stats, report


def visualize_subgraph(G, drug_name: str, depth: int = 2) -> None:
    """Visualise a subgraph centred around a particular drug.

    Parameters
    ----------
    G : nx.DiGraph
        The full knowledge graph.
    drug_name : str
        The drug node to centre the subgraph around.
    depth : int
        How many hops from the drug node to include.

    Notes
    -----
    This function uses Plotly to render an interactive network plot.  It
    traverses breadth‑first from the specified node up to the given depth
    and includes all nodes and edges encountered.  Use this as a quick
    visual inspection tool for the resulting graph.
    """
    # Ensure drug node exists
    if not G.has_node(drug_name):
        raise ValueError(f"Drug '{drug_name}' not found in graph")

    # Breadth first search to collect nodes within specified depth
    visited = {drug_name}
    frontier = {drug_name}
    levels = {drug_name: 0}
    for i in range(1, depth + 1):
        next_frontier = set()
        for node in frontier:
            neighbors = G.successors(node) + G.predecessors(node)
            for nbr in neighbors:
                if nbr not in visited:
                    visited.add(nbr)
                    next_frontier.add(nbr)
                    levels[nbr] = i
        frontier = next_frontier

    subG = G.subgraph(visited)

    # Compute radial layout: nodes at level i are placed on a circle of radius i+1
    # around the origin.  Angle positions are evenly spaced.
    nodes_by_level: Dict[int, List[str]] = defaultdict(list)
    for node, lvl in levels.items():
        nodes_by_level[lvl].append(node)

    pos: Dict[str, Tuple[float, float]] = {}
    for lvl, nodes_at_level in nodes_by_level.items():
        n = len(nodes_at_level)
        # Level 0 is the centre
        if lvl == 0:
            for node in nodes_at_level:
                pos[node] = (0.0, 0.0)
            continue
        # For levels > 0, distribute nodes evenly on a circle of radius equal to level
        radius = float(lvl)
        for j, node in enumerate(nodes_at_level):
            # Avoid division by zero when n == 0 (should not happen)
            angle = 2 * math.pi * j / max(n, 1)
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            pos[node] = (x, y)

    # Build edge traces
    edge_x = []
    edge_y = []
    for u, v, attrs in subG.edges():
        x0, y0 = pos.get(u, (0, 0))
        x1, y1 = pos.get(v, (0, 0))
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1, color='#888'),
                            hoverinfo='none', mode='lines')

    # Build node traces
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    for node in subG.nodes():
        x, y = pos.get(node, (0, 0))
        node_x.append(x)
        node_y.append(y)
        attr = subG._nodes[node]
        node_type = attr.get('type', 'Unknown')
        node_color.append({'Drug': '#1f77b4', 'Disease': '#ff7f0e', 'ICD_Code': '#2ca02c', 'ICD_Root': '#d62728', 'FDA_Approval': '#9467bd'}.get(node_type, '#8c564b'))
        node_text.append(f"{node}\n{node_type}")

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(showscale=False, color=node_color, size=10, line_width=2)
    )

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title=f'Subgraph around {drug_name}',
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                    ))
    fig.show()


__all__ = [
    'build_knowledge_graph',
    'DataQualityReport',
    'visualize_subgraph',
    'export_graph_to_json',
    'export_graph_to_graphml',
]


def export_graph_to_json(G: SimpleDiGraph, filepath: str) -> None:
    """Export the graph to a JSON file.

    The output JSON contains two top‑level keys: ``nodes`` and
    ``edges``.  Each node is represented by a dictionary with an ``id``
    and its attributes; each edge is represented by a dictionary with
    ``source``, ``target`` and its attributes.

    Parameters
    ----------
    G : SimpleDiGraph
        Graph to export.
    filepath : str
        Destination file path for the JSON.
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
        json.dump(data, f, indent=2, default=str)


def export_graph_to_graphml(G: SimpleDiGraph, filepath: str) -> None:
    """Export the graph to a GraphML file.

    GraphML is an XML‑based format for graphs.  This exporter writes
    node and edge attributes as data elements.  All attributes are
    converted to strings; date objects are formatted using ISO format.

    Parameters
    ----------
    G : SimpleDiGraph
        Graph to export.
    filepath : str
        Destination file path.
    """
    import xml.etree.ElementTree as ET

    # Collect all attribute keys from nodes and edges
    node_attr_keys = set()
    for attrs in G.nodes().values():
        node_attr_keys.update(attrs.keys())
    edge_attr_keys = set()
    for _, _, attrs in G.edges():
        edge_attr_keys.update(attrs.keys())

    # Create root element
    root = ET.Element('graphml', xmlns="http://graphml.graphdrawing.org/xmlns")

    # Define keys for node attributes
    key_id_map = {}
    key_counter = 0
    for k in sorted(node_attr_keys):
        kid = f"n{key_counter}"
        key_id_map[k] = kid
        ET.SubElement(root, 'key', id=kid, **{'for': 'node', 'attr.name': k, 'attr.type': 'string'})
        key_counter += 1
    for k in sorted(edge_attr_keys):
        kid = f"e{key_counter}"
        key_id_map[k] = kid
        ET.SubElement(root, 'key', id=kid, **{'for': 'edge', 'attr.name': k, 'attr.type': 'string'})
        key_counter += 1

    # Create graph element
    graph_el = ET.SubElement(root, 'graph', edgedefault='directed')

    # Add nodes
    for node_id, attrs in G.nodes().items():
        node_el = ET.SubElement(graph_el, 'node', id=str(node_id))
        for attr_name, attr_val in attrs.items():
            if attr_name in key_id_map and attr_val is not None:
                # convert value to string
                val_str = str(attr_val)
                data_el = ET.SubElement(node_el, 'data', key=key_id_map[attr_name])
                data_el.text = val_str

    # Add edges
    for idx, (u, v, attrs) in enumerate(G.edges()):
        edge_el = ET.SubElement(graph_el, 'edge', id=f"e{idx}", source=str(u), target=str(v))
        for attr_name, attr_val in attrs.items():
            if attr_name in key_id_map and attr_val is not None:
                val_str = str(attr_val)
                data_el = ET.SubElement(edge_el, 'data', key=key_id_map[attr_name])
                data_el.text = val_str

    # Write to file
    tree = ET.ElementTree(root)
    tree.write(filepath, encoding='utf-8', xml_declaration=True)