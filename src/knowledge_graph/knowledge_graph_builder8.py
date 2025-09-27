#!/usr/bin/env python
"""
Comprehensive pharmaceutical knowledge graph builder.

This module implements an end-to-end pipeline for transforming an RxNav-style
pharmaceutical dataset into a multi-level knowledge graph that captures
relationships between ICD-10 categories, diseases, drugs, RxCUI identifiers and
component metadata (brands, ingredients, precise ingredients, FDA approval
information).

Pipeline phases
---------------
1. Data preparation – loads a CSV/Excel file, standardises text fields, parses
   list-like columns and validates the presence of critical data.
2. ICD-10 processing – extracts ICD-10 codes from structured columns and
   compresses them into chapter-level prefixes (e.g. ``F10-F12``) for use as
   root nodes in the hierarchy.
3. Graph construction – builds a directed ``networkx`` graph with separate node
   types for ICD-10 categories, diseases, drugs, approval dates, RxCUIs, brand
   names, ingredients and precise ingredients.
4. Graph enhancement – enriches nodes with attributes, removes orphaned nodes
   and verifies structural integrity before visualisation.
5. Visualisation & export – produces static (matplotlib, plotly) and
   interactive (PyVis) layouts, exports GraphML/GPickle files, adjacency
   matrices, node/edge tables and a textual summary.
6. Analysis & querying – provides helper functions for querying the graph,
   surfaces high-level insights (ICD clusters, approval timeline, ingredient
   frequency) and stores example query outputs for downstream use.

Usage
-----
Run the script directly and follow the prompts (defaults are provided):

    python knowledge_graph_builder8.py

or supply command-line arguments:

    python knowledge_graph_builder8.py --input data/output/rxnav_with_kg.csv \
        --output_dir data/output/kg

The script will prompt for paths even when CLI arguments are supplied (just hit
*Enter* to accept the default). Set ``--skip_visuals`` to avoid generating
matplotlib/plotly/pyvis artefacts (useful on headless servers).

Requirements
------------
``pandas``, ``networkx``, ``matplotlib``, ``plotly``, ``pyvis``, ``openpyxl``
(as the Excel engine) and the Python standard library modules imported below.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")  # ensure headless rendering works
import matplotlib.pyplot as plt
import networkx as nx
import openpyxl  # noqa: F401 – required for Excel support via pandas
import pandas as pd
import plotly.graph_objects as go
from pyvis.network import Network

# ---------------------------------------------------------------------------
# Constants and configuration
# ---------------------------------------------------------------------------

DEFAULT_INPUT_PATH = "data/output/rxnav_with_kg.xlsx"
DEFAULT_OUTPUT_DIR = "data/output/kg"

CRITICAL_COLUMNS = [
    "Drug Name",
    "RxCUI",
    "Brand Name",
    "Ingredient",
    "Precise Ingredient",
    "may_treat_diseases",
    "may_treat_diseases_icd10",
    "fda_approval_date",
]

LIST_COLUMNS = ["Brand Name", "Ingredient", "Precise Ingredient", "may_treat_diseases"]

TYPE_COLOURS = {
    "Drug": "#1f77b4",
    "Disease": "#ff7f0e",
    "ICD10Category": "#9467bd",
    "Approval_Date": "#2ca02c",
    "RxCUI": "#17becf",
    "Brand": "#8c564b",
    "Ingredient": "#bcbd22",
    "Precise_Ingredient": "#7f7f7f",
}

ICD_LETTER_CHAPTERS = {
    "A": "Certain infectious and parasitic diseases",
    "B": "Certain infectious and parasitic diseases",
    "C": "Neoplasms",
    "D": "Diseases of the blood and blood-forming organs",
    "E": "Endocrine, nutritional and metabolic diseases",
    "F": "Mental, behavioural and neurodevelopmental disorders",
    "G": "Diseases of the nervous system",
    "H": "Diseases of the eye, adnexa, ear and mastoid process",
    "I": "Diseases of the circulatory system",
    "J": "Diseases of the respiratory system",
    "K": "Diseases of the digestive system",
    "L": "Diseases of the skin and subcutaneous tissue",
    "M": "Diseases of the musculoskeletal system and connective tissue",
    "N": "Diseases of the genitourinary system",
    "O": "Pregnancy, childbirth and the puerperium",
    "P": "Certain conditions originating in the perinatal period",
    "Q": "Congenital malformations, deformations and chromosomal abnormalities",
    "R": "Symptoms, signs and abnormal clinical findings",
    "S": "Injury, poisoning and certain other consequences of external causes",
    "T": "Injury, poisoning and certain other consequences of external causes",
    "V": "External causes of morbidity",
    "W": "External causes of morbidity",
    "X": "External causes of morbidity",
    "Y": "External causes of morbidity",
    "Z": "Factors influencing health status and contact with health services",
}

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def console_header(message: str) -> None:
    print("\n" + "=" * 80)
    print(message)
    print("=" * 80)


def clean_text(value: Optional[object]) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def to_list(value: Optional[object]) -> List[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return [clean_text(v) for v in value if clean_text(v)]
    if isinstance(value, (tuple, set)):
        return [clean_text(v) for v in value if clean_text(v)]

    text = clean_text(value)
    if not text or text.lower() in {"nan", "none"}:
        return []

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple, set)):
            return [clean_text(v) for v in parsed if clean_text(v)]
    except Exception:
        pass

    parts = re.split(r"[|;,]\s*", text)
    return [p.strip() for p in parts if p.strip() and p.strip().lower() not in {"nan", "none"}]


def parse_may_treat_icd10(cell: Optional[object]) -> Dict[str, List[str]]:
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return {}

    text = clean_text(cell)
    if not text:
        return {}

    try:
        parsed = ast.literal_eval(text)
    except Exception:
        return {}

    merged: Dict[str, List[str]] = defaultdict(list)

    if isinstance(parsed, dict):
        items = parsed.items()
    elif isinstance(parsed, list):
        items = []
        for candidate in parsed:
            if isinstance(candidate, dict):
                items.extend(candidate.items())
    else:
        items = []

    for disease, codes in items:
        disease_name = clean_text(disease)
        if not disease_name:
            continue
        for code in to_list(codes):
            normalised = re.sub(r"[^A-Za-z0-9.]", "", code).upper()
            if normalised and normalised not in merged[disease_name]:
                merged[disease_name].append(normalised)

    return dict(merged)


def icd_prefix(code: str) -> str:
    return clean_text(code).split(".")[0].upper()


def compress_continuous_prefixes(prefixes: Iterable[str]) -> List[str]:
    normalised: List[str] = []
    seen: set[str] = set()
    for prefix in prefixes:
        cleaned = clean_text(prefix).upper()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        normalised.append(cleaned)

    numeric_tokens: List[Tuple[str, int]] = []
    passthrough: List[str] = []
    for token in normalised:
        match = re.match(r"^([A-Z]+)(\d+)$", token)
        if match:
            numeric_tokens.append((match.group(1), int(match.group(2))))
        else:
            passthrough.append(token)

    numeric_tokens.sort(key=lambda item: (item[0], item[1]))
    compressed: List[str] = []
    idx = 0
    while idx < len(numeric_tokens):
        letter, start = numeric_tokens[idx]
        end = start
        idx += 1
        while idx < len(numeric_tokens) and numeric_tokens[idx][0] == letter and numeric_tokens[idx][1] == end + 1:
            end = numeric_tokens[idx][1]
            idx += 1
        if start == end:
            compressed.append(f"{letter}{start}")
        else:
            compressed.append(f"{letter}{start}-{letter}{end}")

    for token in normalised:
        if token not in {f"{letter}{num}" for letter, num in numeric_tokens} and token not in compressed:
            compressed.append(token)

    return compressed


def disease_to_icd_ranges(mapping: Dict[str, List[str]]) -> Dict[str, List[str]]:
    output: Dict[str, List[str]] = {}
    for disease, codes in mapping.items():
        prefixes = [icd_prefix(code) for code in codes if clean_text(code)]
        output[disease] = compress_continuous_prefixes(prefixes)
    return output


def parse_approval_date(value: Optional[object]) -> Optional[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    text = clean_text(value)
    if not text:
        return None
    formats = ["%Y-%m-%d", "%m/%d/%Y", "%Y/%m/%d", "%d-%b-%Y", "%b %d, %Y"]
    for fmt in formats:
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            continue
    try:
        parsed = pd.to_datetime(text, errors="coerce")
        if pd.notna(parsed):
            return parsed.date().isoformat()
    except Exception:
        pass
    return None


def icd_category_description(code: str) -> str:
    if not code:
        return ""
    letter = code[0].upper()
    return ICD_LETTER_CHAPTERS.get(letter, "Unknown category")


def load_dataset(path: str) -> pd.DataFrame:
    resolved = os.path.abspath(path)
    if not os.path.exists(resolved):
        raise FileNotFoundError(f"Input file not found: {resolved}")

    _, ext = os.path.splitext(resolved)
    ext = ext.lower()
    if ext in {".xlsx", ".xls"}:
        df = pd.read_excel(resolved, engine="openpyxl")
    elif ext == ".csv":
        df = pd.read_csv(resolved)
    else:
        raise ValueError(f"Unsupported input format: {ext}")

    df = df.rename(columns={col: clean_text(col) for col in df.columns})
    return df


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for column in CRITICAL_COLUMNS:
        if column not in df.columns:
            df[column] = pd.NA
    return df


def standardise_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = ensure_columns(df)
    df = df.copy()

    for column in LIST_COLUMNS:
        df[column] = df[column].apply(to_list)

    if "Drug Name" in df.columns:
        df["Drug Name"] = df["Drug Name"].apply(clean_text)
    if "RxCUI" in df.columns:
        df["RxCUI"] = df["RxCUI"].apply(clean_text)
    if "may_treat_diseases" in df.columns:
        df["may_treat_diseases"] = df["may_treat_diseases"].apply(to_list)

    return df


def validate_dataframe(df: pd.DataFrame) -> None:
    console_header("Phase 1 – Data Validation")
    for column in CRITICAL_COLUMNS:
        if column not in df.columns:
            print(f"[WARN] Missing expected column: {column}")
            continue
        missing = int(df[column].isna().sum())
        print(f"{column}: {missing} missing values")
    print(f"Total rows: {len(df)}")


# ---------------------------------------------------------------------------
# Filtering utilities
# ---------------------------------------------------------------------------


def filter_by_drugs(df: pd.DataFrame, drugs: Sequence[str]) -> pd.DataFrame:
    if not drugs:
        return df.copy()
    normalised = {drug.lower().strip() for drug in drugs if drug}
    mask = df["Drug Name"].astype(str).str.lower().isin(normalised)
    filtered = df[mask].copy()
    return filtered


def save_filtered_dataset(df: pd.DataFrame, output_dir: str) -> Optional[str]:
    if df.empty:
        print("[WARN] Filter produced an empty dataset; skipping filtered export")
        return None
    path = os.path.join(output_dir, "filtered_dataset.csv")
    df.to_csv(path, index=False)
    print(f"Filtered dataset saved to: {path}")
    return path


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def add_or_merge_node(graph: nx.DiGraph, node_id: str, **attrs: object) -> None:
    if not graph.has_node(node_id):
        graph.add_node(node_id, **attrs)
    else:
        existing = graph.nodes[node_id]
        for key, value in attrs.items():
            if value is None or value == "":
                continue
            if key not in existing or not existing[key]:
                existing[key] = value
            elif isinstance(existing.get(key), list) and isinstance(value, list):
                merged = existing[key]
                for item in value:
                    if item not in merged:
                        merged.append(item)
                existing[key] = merged
            elif isinstance(existing.get(key), set) and isinstance(value, set):
                existing[key] = existing[key] | value
            else:
                existing[key] = existing.get(key)


def build_graph(df: pd.DataFrame) -> nx.DiGraph:
    console_header("Phase 3 – Building knowledge graph")
    graph = nx.DiGraph(name="Pharmaceutical Knowledge Graph")

    for idx, row in df.iterrows():
        drug_name = clean_text(row.get("Drug Name"))
        if not drug_name:
            continue

        drug_id = f"Drug::{drug_name}"
        rxcui = clean_text(row.get("RxCUI"))
        approval_iso = parse_approval_date(row.get("fda_approval_date"))
        brand_names = to_list(row.get("Brand Name"))
        ingredients = to_list(row.get("Ingredient"))
        precise_ings = to_list(row.get("Precise Ingredient"))

        drug_attrs = {
            "label": drug_name,
            "type": "Drug",
            "level": 4,
            "rxcui": rxcui,
            "approval_date": approval_iso,
            "brand_names": brand_names,
            "ingredients": ingredients,
            "precise_ingredients": precise_ings,
        }
        add_or_merge_node(graph, drug_id, **drug_attrs)

        if approval_iso:
            date_id = f"ApprovalDate::{approval_iso}"
            add_or_merge_node(
                graph,
                date_id,
                label=approval_iso,
                type="Approval_Date",
                level=4,
            )
            graph.add_edge(drug_id, date_id, type="APPROVED_ON")

        if rxcui:
            rxcui_id = f"RxCUI::{rxcui}"
            add_or_merge_node(
                graph,
                rxcui_id,
                label=rxcui,
                type="RxCUI",
                level=4,
            )
            graph.add_edge(drug_id, rxcui_id, type="HAS_RXCUI")

            for brand in brand_names:
                brand_id = f"Brand::{brand}"
                add_or_merge_node(
                    graph,
                    brand_id,
                    label=brand,
                    type="Brand",
                    level=5,
                )
                graph.add_edge(rxcui_id, brand_id, type="BRANDED_AS")

            for ingredient in ingredients:
                ingredient_id = f"Ingredient::{ingredient}"
                add_or_merge_node(
                    graph,
                    ingredient_id,
                    label=ingredient,
                    type="Ingredient",
                    level=5,
                )
                graph.add_edge(rxcui_id, ingredient_id, type="CONTAINS")

            for precise in precise_ings:
                precise_id = f"PreciseIngredient::{precise}"
                add_or_merge_node(
                    graph,
                    precise_id,
                    label=precise,
                    type="Precise_Ingredient",
                    level=5,
                )
                graph.add_edge(rxcui_id, precise_id, type="CONTAINS")

        disease_mapping_raw = parse_may_treat_icd10(row.get("may_treat_diseases_icd10"))
        disease_mapping_ranges = disease_to_icd_ranges(disease_mapping_raw)

        diseases_from_alt_col = to_list(row.get("may_treat_diseases"))
        if not disease_mapping_ranges and diseases_from_alt_col:
            disease_mapping_ranges = {disease: [] for disease in diseases_from_alt_col}

        for disease, icd_ranges in disease_mapping_ranges.items():
            disease_name = clean_text(disease)
            if not disease_name:
                continue
            disease_id = f"Disease::{disease_name}"
            add_or_merge_node(
                graph,
                disease_id,
                label=disease_name,
                type="Disease",
                level=3,
                icd_codes=sorted(icd_ranges) if icd_ranges else [],
            )
            graph.add_edge(drug_id, disease_id, type="TREATS")

            for icd_code in icd_ranges:
                icd_label = clean_text(icd_code).upper()
                if not icd_label:
                    continue
                icd_id = f"ICD10::{icd_label}"
                add_or_merge_node(
                    graph,
                    icd_id,
                    label=icd_label,
                    type="ICD10Category",
                    level=1,
                    code=icd_label,
                    code_category=icd_category_description(icd_label),
                )
                graph.add_edge(disease_id, icd_id, type="DIAGNOSED_AS")

    print(f"Graph constructed with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
    return graph


# ---------------------------------------------------------------------------
# Graph validation and enhancement
# ---------------------------------------------------------------------------


def remove_orphan_nodes(graph: nx.DiGraph) -> List[str]:
    isolates = list(nx.isolates(graph))
    if isolates:
        print(f"Removing {len(isolates)} orphaned nodes from the graph")
        graph.remove_nodes_from(isolates)
    return isolates


def ensure_level_attributes(graph: nx.DiGraph) -> None:
    for node, data in graph.nodes(data=True):
        if "level" not in data:
            data["level"] = 0


def validate_graph(graph: nx.DiGraph) -> None:
    console_header("Phase 4 – Graph validation")
    ensure_level_attributes(graph)
    removed = remove_orphan_nodes(graph)
    if not removed:
        print("No orphaned nodes detected")
    print(f"Final node count: {graph.number_of_nodes()}")
    print(f"Final edge count: {graph.number_of_edges()}")


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------


def serialise_attribute(value: object) -> object:
    if isinstance(value, (list, tuple, set)):
        return "|".join(str(v) for v in value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return value


def graph_for_graphml(graph: nx.DiGraph) -> nx.DiGraph:
    transformed = nx.DiGraph()
    for node, attrs in graph.nodes(data=True):
        transformed.add_node(node, **{k: serialise_attribute(v) for k, v in attrs.items()})
    for src, dst, attrs in graph.edges(data=True):
        transformed.add_edge(src, dst, **{k: serialise_attribute(v) for k, v in attrs.items()})
    return transformed


def export_graph(graph: nx.DiGraph, output_dir: str) -> Dict[str, str]:
    console_header("Phase 5 – Exporting graph artefacts")
    os.makedirs(output_dir, exist_ok=True)

    paths: Dict[str, str] = {}
    graphml_path = os.path.join(output_dir, "knowledge_graph.graphml")
    gpickle_path = os.path.join(output_dir, "knowledge_graph.gpickle")
    nodes_csv_path = os.path.join(output_dir, "nodes.csv")
    edges_csv_path = os.path.join(output_dir, "edges.csv")
    adjacency_path = os.path.join(output_dir, "adjacency_matrix.csv")
    type_adjacency_path = os.path.join(output_dir, "type_adjacency_matrix.csv")
    summary_path = os.path.join(output_dir, "summary.txt")

    graphml_ready = graph_for_graphml(graph)
    nx.write_graphml(graphml_ready, graphml_path)
    paths["graphml"] = graphml_path
    with open(gpickle_path, "wb") as fh:
        import pickle
        pickle.dump(graph, fh)
    paths["gpickle"] = gpickle_path

    nodes_records = []
    for node, attrs in graph.nodes(data=True):
        record = {"id": node}
        record.update({k: serialise_attribute(v) for k, v in attrs.items()})
        nodes_records.append(record)
    nodes_df = pd.DataFrame(nodes_records)
    nodes_df.to_csv(nodes_csv_path, index=False)
    paths["nodes_csv"] = nodes_csv_path

    edges_records = []
    for src, dst, attrs in graph.edges(data=True):
        record = {"src": src, "dst": dst}
        record.update({k: serialise_attribute(v) for k, v in attrs.items()})
        edges_records.append(record)
    edges_df = pd.DataFrame(edges_records)
    edges_df.to_csv(edges_csv_path, index=False)
    paths["edges_csv"] = edges_csv_path

    adjacency_df = nx.to_pandas_adjacency(graph, dtype=int, weight=None)
    adjacency_df.to_csv(adjacency_path)
    paths["adjacency_csv"] = adjacency_path

    type_adj_counter: Dict[Tuple[str, str, str], int] = defaultdict(int)
    for src, dst, attrs in graph.edges(data=True):
        src_type = graph.nodes[src].get("type", "")
        dst_type = graph.nodes[dst].get("type", "")
        rel_type = attrs.get("type", "")
        type_adj_counter[(src_type, rel_type, dst_type)] += 1
    type_adj_df = pd.DataFrame(
        [
            {"source_type": s, "relationship": r, "target_type": t, "count": count}
            for (s, r, t), count in sorted(type_adj_counter.items())
        ]
    )
    type_adj_df.to_csv(type_adjacency_path, index=False)
    paths["type_adjacency_csv"] = type_adjacency_path

    node_types = Counter(nx.get_node_attributes(graph, "type").values())
    edge_types = Counter(attrs.get("type", "") for _, _, attrs in graph.edges(data=True))
    with open(summary_path, "w", encoding="utf-8") as fh:
        fh.write("Pharmaceutical Knowledge Graph Summary\n")
        fh.write("-" * 50 + "\n")
        fh.write(f"Nodes: {graph.number_of_nodes()}\n")
        fh.write(f"Edges: {graph.number_of_edges()}\n\n")
        fh.write("Nodes by type:\n")
        for node_type, count in sorted(node_types.items(), key=lambda item: (-item[1], item[0])):
            fh.write(f"  {node_type}: {count}\n")
        fh.write("\nEdges by relationship:\n")
        for rel_type, count in sorted(edge_types.items(), key=lambda item: (-item[1], item[0])):
            fh.write(f"  {rel_type}: {count}\n")
    paths["summary"] = summary_path

    print("Artefacts written:")
    for label, path in paths.items():
        print(f"  {label}: {path}")
    return paths


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------


def hierarchical_positions(graph: nx.DiGraph) -> Dict[str, Tuple[float, float]]:
    levels: Dict[int, List[str]] = defaultdict(list)
    for node, data in graph.nodes(data=True):
        level = int(data.get("level", 0))
        levels[level].append(node)

    positions: Dict[str, Tuple[float, float]] = {}
    for level, nodes in sorted(levels.items()):
        count = len(nodes)
        if count == 0:
            continue
        x_spacing = 4.0 / max(count - 1, 1)
        x_start = -2.0
        for idx, node in enumerate(sorted(nodes)):
            x_coord = x_start + idx * x_spacing if count > 1 else 0.0
            y_coord = -level
            positions[node] = (x_coord, y_coord)
    return positions


def plot_matplotlib_hierarchy(graph: nx.DiGraph, output_dir: str, positions: Dict[str, Tuple[float, float]]) -> Optional[str]:
    if not graph.nodes:
        return None
    path = os.path.join(output_dir, "hierarchy_matplotlib.png")
    colours = [TYPE_COLOURS.get(data.get("type"), "#999999") for _, data in graph.nodes(data=True)]
    plt.figure(figsize=(12, 8))
    nx.draw_networkx_edges(graph, positions, arrows=True, alpha=0.3, width=0.8)
    nx.draw_networkx_nodes(graph, positions, node_color=colours, node_size=160)
    labels = {node: data.get("label", node).split("::")[-1] for node, data in graph.nodes(data=True)}
    nx.draw_networkx_labels(graph, positions, labels, font_size=8)
    plt.title("Pharmaceutical Knowledge Graph – Hierarchical Layout")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Matplotlib hierarchy saved to: {path}")
    return path


def plot_plotly_hierarchy(graph: nx.DiGraph, output_dir: str, positions: Dict[str, Tuple[float, float]]) -> Optional[str]:
    if not graph.nodes:
        return None
    path = os.path.join(output_dir, "hierarchy_plotly.html")

    edge_x: List[float] = []
    edge_y: List[float] = []
    for src, dst in graph.edges():
        x0, y0 = positions.get(src, (0.0, 0.0))
        x1, y1 = positions.get(dst, (0.0, 0.0))
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    node_x = []
    node_y = []
    texts = []
    colours = []
    for node, data in graph.nodes(data=True):
        x, y = positions.get(node, (0.0, 0.0))
        node_x.append(x)
        node_y.append(y)
        label = data.get("label", node)
        type_name = data.get("type", "")
        texts.append(f"{label}<br>Type: {type_name}")
        colours.append(TYPE_COLOURS.get(type_name, "#888888"))

    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=1, color="#BBBBBB"), hoverinfo="none")
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        marker=dict(color=colours, size=12, line=dict(width=0.5, color="#333333")),
        text=texts,
        hoverinfo="text",
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title="Pharmaceutical Knowledge Graph – Hierarchical Layout",
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    fig.write_html(path, include_plotlyjs="cdn")
    print(f"Plotly hierarchy saved to: {path}")
    return path


def export_pyvis(graph: nx.DiGraph, output_dir: str) -> Optional[str]:
    if not graph.nodes:
        return None
    path = os.path.join(output_dir, "knowledge_graph_pyvis.html")
    net_graph = Network(height="750px", width="100%", directed=True)
    net_graph.force_atlas_2based(gravity=-40)
    for node, data in graph.nodes(data=True):
        label = data.get("label", node)
        node_type = data.get("type", "")
        title_parts = [f"<b>{label}</b>", f"Type: {node_type}"]
        for key, value in data.items():
            if key in {"label", "type", "level"}:
                continue
            if value:
                title_parts.append(f"{key}: {serialise_attribute(value)}")
        colour = TYPE_COLOURS.get(node_type, "#CCCCCC")
        net_graph.add_node(node, label=label, title="<br>".join(title_parts), color=colour, level=data.get("level", 0))
    for src, dst, attrs in graph.edges(data=True):
        label = attrs.get("type", "")
        net_graph.add_edge(src, dst, label=label, arrows="to")
    net_graph.set_options(
        """
        var options = {
          "interaction": {"hover": true},
          "physics": {"stabilization": true}
        }
        """
    )
    net_graph.write_html(path)
    print(f"PyVis network saved to: {path}")
    return path


def generate_visuals(graph: nx.DiGraph, output_dir: str) -> None:
    console_header("Phase 5 – Visualisation")
    os.makedirs(output_dir, exist_ok=True)
    positions = hierarchical_positions(graph)
    plot_matplotlib_hierarchy(graph, output_dir, positions)
    plot_plotly_hierarchy(graph, output_dir, positions)
    export_pyvis(graph, output_dir)


# ---------------------------------------------------------------------------
# Query functions
# ---------------------------------------------------------------------------


def find_node_by_label(graph: nx.DiGraph, label: str, node_type: str) -> Optional[str]:
    label_lower = label.lower().strip()
    for node, data in graph.nodes(data=True):
        if data.get("type") == node_type and data.get("label", "").lower() == label_lower:
            return node
    return None


def find_diseases_by_drug(graph: nx.DiGraph, drug_name: str) -> List[str]:
    node_id = find_node_by_label(graph, drug_name, "Drug")
    if not node_id:
        return []
    diseases = [graph.nodes[dst].get("label", dst) for _, dst, attrs in graph.out_edges(node_id, data=True) if attrs.get("type") == "TREATS"]
    return sorted(set(diseases))


def find_drugs_by_icd(graph: nx.DiGraph, icd_code: str) -> List[str]:
    code_upper = clean_text(icd_code).upper()
    targets = []
    for node, data in graph.nodes(data=True):
        if data.get("type") != "ICD10Category":
            continue
        label = data.get("label", "").upper()
        if label == code_upper or code_upper in label:
            incoming = graph.in_edges(node, data=True)
            diseases = [src for src, _, attrs in incoming if attrs.get("type") == "DIAGNOSED_AS"]
            drug_names: set[str] = set()
            for disease in diseases:
                for src, _, attrs in graph.in_edges(disease, data=True):
                    if attrs.get("type") == "TREATS":
                        drug_names.add(graph.nodes[src].get("label", src))
            targets.append((label, sorted(drug_names)))
    aggregated: set[str] = set()
    for _, drug_list in targets:
        aggregated.update(drug_list)
    return sorted(aggregated)


def get_drug_ingredient_relationships(graph: nx.DiGraph, drug_name: str) -> Dict[str, List[str]]:
    node_id = find_node_by_label(graph, drug_name, "Drug")
    if not node_id:
        return {"Brand": [], "Ingredient": [], "Precise_Ingredient": []}
    result: Dict[str, set[str]] = {"Brand": set(), "Ingredient": set(), "Precise_Ingredient": set()}
    for _, rxcui_node, attrs in graph.out_edges(node_id, data=True):
        if attrs.get("type") != "HAS_RXCUI":
            continue
        for _, child, edge_attrs in graph.out_edges(rxcui_node, data=True):
            rel_type = edge_attrs.get("type")
            target_type = graph.nodes[child].get("type")
            if rel_type == "BRANDED_AS" and target_type == "Brand":
                result["Brand"].add(graph.nodes[child].get("label", child))
            if rel_type == "CONTAINS" and target_type == "Ingredient":
                result["Ingredient"].add(graph.nodes[child].get("label", child))
            if rel_type == "CONTAINS" and target_type == "Precise_Ingredient":
                result["Precise_Ingredient"].add(graph.nodes[child].get("label", child))
    return {key: sorted(values) for key, values in result.items()}


# ---------------------------------------------------------------------------
# Insight generation
# ---------------------------------------------------------------------------


def most_connected_icd_categories(graph: nx.DiGraph, top_n: int = 10) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for node, data in graph.nodes(data=True):
        if data.get("type") != "ICD10Category":
            continue
        diseases = {
            src
            for src, _, attrs in graph.in_edges(node, data=True)
            if attrs.get("type") == "DIAGNOSED_AS"
        }
        records.append(
            {
                "code": data.get("label", node),
                "description": data.get("code_category", ""),
                "disease_count": len(diseases),
            }
        )
    records.sort(key=lambda item: (-item["disease_count"], item["code"]))
    return records[:top_n]


def drug_approval_timeline(df: pd.DataFrame) -> List[Dict[str, object]]:
    counter = Counter()
    if "fda_approval_date" not in df.columns:
        return []
    for value in df["fda_approval_date"]:
        iso = parse_approval_date(value)
        if iso:
            counter[iso[:4]] += 1
    return [
        {"year": year, "approvals": counter[year]}
        for year in sorted(counter)
    ]


def ingredient_frequency_analysis(df: pd.DataFrame, top_n: int = 15) -> List[Dict[str, object]]:
    counter = Counter()
    if "Ingredient" not in df.columns:
        return []
    for value in df["Ingredient"]:
        for ingredient in to_list(value):
            counter[ingredient] += 1
    most_common = counter.most_common(top_n)
    return [
        {"ingredient": ingredient, "frequency": frequency}
        for ingredient, frequency in most_common
    ]


def generate_insights(graph: nx.DiGraph, df: pd.DataFrame) -> Dict[str, object]:
    insights = {
        "most_connected_icd_categories": most_connected_icd_categories(graph),
        "drug_approval_timeline": drug_approval_timeline(df),
        "ingredient_frequency": ingredient_frequency_analysis(df),
    }
    return insights


def write_insights(insights: Dict[str, object], output_dir: str) -> str:
    path = os.path.join(output_dir, "insights.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(insights, fh, indent=2)
    print(f"Insights saved to: {path}")
    return path


# ---------------------------------------------------------------------------
# Query demonstration
# ---------------------------------------------------------------------------


def run_sample_queries(
    graph: nx.DiGraph,
    df: pd.DataFrame,
    output_dir: str,
    filtered_drug_names: Sequence[str],
) -> Optional[str]:
    console_header("Phase 6 – Query demonstrations")
    samples: Dict[str, object] = {}

    candidate_drugs: List[str] = []
    if filtered_drug_names:
        candidate_drugs.extend(filtered_drug_names)
    if not candidate_drugs:
        for value in df["Drug Name"]:
            name = clean_text(value)
            if name:
                candidate_drugs.append(name)
                break

    drug_results = {}
    for drug in candidate_drugs[:3]:
        diseases = find_diseases_by_drug(graph, drug)
        components = get_drug_ingredient_relationships(graph, drug)
        drug_results[drug] = {
            "diseases": diseases,
            "components": components,
        }
        print(f"Sample drug '{drug}' treats {len(diseases)} diseases")

    samples["drug_profiles"] = drug_results

    icd_sample = None
    for node, data in graph.nodes(data=True):
        if data.get("type") == "ICD10Category":
            icd_sample = data.get("label", node)
            break
    if icd_sample:
        drugs_for_icd = find_drugs_by_icd(graph, icd_sample)
        samples["icd_lookup"] = {"code": icd_sample, "drugs": drugs_for_icd}
        print(f"ICD-10 category '{icd_sample}' has {len(drugs_for_icd)} associated drugs")

    path = os.path.join(output_dir, "query_samples.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(samples, fh, indent=2)
    print(f"Query samples written to: {path}")
    return path


# ---------------------------------------------------------------------------
# CLI orchestration
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and analyse a pharmaceutical knowledge graph")
    parser.add_argument("--input", help="Path to CSV/Excel dataset", default=None)
    parser.add_argument("--output_dir", help="Directory for exports", default=None)
    parser.add_argument("--drug", action="append", help="Drug name to filter (can be supplied multiple times)")
    parser.add_argument("--skip_visuals", action="store_true", help="Skip generation of matplotlib/plotly/pyvis visuals")
    return parser.parse_args()


def prompt_with_default(prompt: str, default: str) -> str:
    response = input(f"{prompt} [{default}]: ").strip()
    return response or default


def main() -> None:
    args = parse_args()

    console_header("Knowledge Graph Builder")
    input_path_default = args.input or DEFAULT_INPUT_PATH
    output_dir_default = args.output_dir or DEFAULT_OUTPUT_DIR

    input_path = prompt_with_default("Enter path to Excel/CSV file", input_path_default)
    output_dir = prompt_with_default("Enter output directory", output_dir_default)
    drug_prompt = "Enter drug names to filter (comma separated, leave blank for all)"
    drug_input = prompt_with_default(drug_prompt, ", ".join(args.drug) if args.drug else "").strip()
    filter_drugs = [clean_text(item) for item in drug_input.split(",") if clean_text(item)]

    print("\nStarting pipeline execution...")

    df = load_dataset(input_path)
    validate_dataframe(df)
    df = standardise_dataframe(df)

    os.makedirs(output_dir, exist_ok=True)

    filtered_df = filter_by_drugs(df, filter_drugs)
    if filter_drugs:
        save_filtered_dataset(filtered_df, output_dir)

    graph = build_graph(df)
    validate_graph(graph)

    export_graph(graph, output_dir)

    if args.skip_visuals:
        print("Visual generation skipped via --skip_visuals")
    else:
        generate_visuals(graph, output_dir)

    insights = generate_insights(graph, df)
    write_insights(insights, output_dir)

    run_sample_queries(graph, df, output_dir, filter_drugs)

    print("\nPipeline complete. Review the output directory for artefacts and reports.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAborted by user")
        sys.exit(1)
