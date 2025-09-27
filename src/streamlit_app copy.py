"""Streamlit Drug Knowledge Graph Explorer.

This Streamlit application provides an interactive interface for exploring the
pharmaceutical knowledge graph produced by ``knowledge_graph_builder8.py``. The
app supports drug-centric, disease-centric, ICD-10, ingredient, approval date
and advanced multi-parameter queries, alongside rich visual analytics and data
export utilities.
"""

from __future__ import annotations

import base64
import json
import pickle
import re
import zipfile
from datetime import date, datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ---------------------------------------------------------------------------
# Constants and configuration
# ---------------------------------------------------------------------------

DEFAULT_GRAPH_CANDIDATES = [
    Path("data/output/kg/knowledge_graph.gpickle"),
    Path("data/output/kg/knowledge_graph.graphml"),
    Path("src/knowledge_graph/knowledge_graph.gpickle"),
    Path("src/knowledge_graph/drug_graph.graphml"),
]

GRAPH_LIST_FIELDS = {"brand_names", "ingredients", "precise_ingredients"}
GRAPH_TYPE_FIELDS = {"type", "label"}

PAGE_NAMES = [
    "Home",
    "Drug Lookup",
    "Disease Explorer",
    "ICD-10 Navigator",
    "Advanced Search",
    "Graph Visualization",
    "Data Export",
    "Analytics Dashboard",
]

ICD_CODE_PATTERN = re.compile(r"^[A-TV-Z][0-9A-Z]{0,6}$")
MAX_HISTORY_ENTRIES = 50
APPROVAL_DATE_FIELDS = ("fda_approval_date", "approval_date")

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def normalise_text(value: str) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip().lower()


def display_label(node_id: str, data: Dict[str, Any]) -> str:
    label = data.get("label") or node_id.split("::")[-1]
    return label.replace("\n", " ").strip()


def ensure_iterable(value: Any) -> List[str]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, tuple) or isinstance(value, set):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        parts = [part.strip() for part in value.replace(";", "|").split("|")]
        return [part for part in parts if part]
    return [str(value).strip()]


def parse_iso_date(value: Any) -> Optional[date]:
    if value is None or value == "":
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    text = str(value).strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%m/%d/%Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(text).date()
    except ValueError:
        return None


def resolve_approval_value(data: Dict[str, Any]) -> Optional[str]:
    for field in APPROVAL_DATE_FIELDS:
        value = data.get(field)
        if value is None:
            continue
        text = str(value).strip()
        if text and text.lower() not in {"unknown", "nan", "none"}:
            return text
    return None


def find_default_graph_path() -> Optional[str]:
    for candidate in DEFAULT_GRAPH_CANDIDATES:
        if candidate.exists():
            return str(candidate)
    return None


def ensure_graph_schema(graph: nx.DiGraph) -> nx.DiGraph:
    for node, data in graph.nodes(data=True):
        if "label" not in data or not data["label"]:
            data["label"] = node.split("::")[-1]
        for field in GRAPH_LIST_FIELDS:
            data[field] = ensure_iterable(data.get(field))
        if data.get("type") == "Drug":
            existing = str(data.get("approval_date", "")).strip()
            if not existing or existing.lower() in {"unknown", "nan", "none"}:
                resolved = resolve_approval_value(data)
                if resolved:
                    parsed = parse_iso_date(resolved)
                    data["approval_date"] = parsed.isoformat() if parsed else resolved
    for src, dst, attrs in graph.edges(data=True):
        if "type" not in attrs and "relation" in attrs:
            attrs["type"] = attrs["relation"]
    return graph


def get_graph_signature(graph: nx.DiGraph, source_path: str) -> str:
    stats = f"{source_path}:{graph.number_of_nodes()}:{graph.number_of_edges()}"
    return stats


# ---------------------------------------------------------------------------
# Streamlit caching helpers
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def load_knowledge_graph(file_path: str) -> nx.DiGraph:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Graph file not found at {file_path}")
    suffix = path.suffix.lower()
    if suffix in {".gpickle", ".pickle", ".pkl"}:
        with open(path, "rb") as handle:
            graph = pickle.load(handle)
    elif suffix in {".graphml", ".xml"}:
        graph = nx.read_graphml(path)
    else:
        raise ValueError("Supported graph formats are .gpickle and .graphml")
    if not isinstance(graph, nx.DiGraph):
        graph = graph.to_directed()
    return ensure_graph_schema(graph)


@st.cache_data(show_spinner=False)
def get_all_nodes_by_type(graph_path: str, node_type: str) -> List[str]:
    graph = load_knowledge_graph(graph_path)
    labels = [display_label(node, data) for node, data in graph.nodes(data=True) if data.get("type") == node_type]
    return sorted(set(labels))


@st.cache_data(show_spinner=False)
def get_all_drugs(graph_path: str) -> List[str]:
    return get_all_nodes_by_type(graph_path, "Drug")


@st.cache_data(show_spinner=False)
def get_all_diseases(graph_path: str) -> List[str]:
    return get_all_nodes_by_type(graph_path, "Disease")


@st.cache_data(show_spinner=False)
def get_all_icd10_codes(graph_path: str) -> List[str]:
    return get_all_nodes_by_type(graph_path, "ICD10Category")


@st.cache_data(show_spinner=False)
def build_degree_dataframe(graph_path: str) -> pd.DataFrame:
    graph = load_knowledge_graph(graph_path)
    records = []
    for node, data in graph.nodes(data=True):
        node_type = data.get("type", "Unknown")
        records.append(
            {
                "node": display_label(node, data),
                "type": node_type,
                "in_degree": graph.in_degree(node),
                "out_degree": graph.out_degree(node),
            }
        )
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Graph lookup utilities
# ---------------------------------------------------------------------------


def find_node(graph: nx.DiGraph, name: str, node_type: str) -> Optional[str]:
    target = normalise_text(name)
    for node, data in graph.nodes(data=True):
        if data.get("type") != node_type:
            continue
        label = display_label(node, data)
        if normalise_text(label) == target:
            return node
    return None


def fuzzy_search_candidates(graph_path: str, query: str, node_type: str, limit: int = 5) -> List[str]:
    from difflib import get_close_matches

    if not query:
        return []
    options = get_all_nodes_by_type(graph_path, node_type)
    matches = get_close_matches(query, options, n=limit, cutoff=0.6)
    return matches


# ---------------------------------------------------------------------------
# Core query implementations
# ---------------------------------------------------------------------------


def _query_drug(graph: nx.DiGraph, drug_name: str) -> Dict[str, Any]:
    node_id = find_node(graph, drug_name, "Drug")
    if not node_id:
        return {}
    data = graph.nodes[node_id]
    approval_raw = resolve_approval_value(data)
    approval_value = None
    if approval_raw:
        parsed = parse_iso_date(approval_raw)
        approval_value = parsed.isoformat() if parsed else approval_raw
    diseases: List[str] = []
    icd_codes: List[str] = []
    icd_categories: List[str] = []
    icd_records: List[Dict[str, str]] = []
    for _, disease_node, attrs in graph.out_edges(node_id, data=True):
        if attrs.get("type") != "TREATS":
            continue
        disease_data = graph.nodes[disease_node]
        disease_label = display_label(disease_node, disease_data)
        diseases.append(disease_label)
        for _, icd_node, icd_attrs in graph.out_edges(disease_node, data=True):
            if icd_attrs.get("type") != "DIAGNOSED_AS":
                continue
            icd_data = graph.nodes[icd_node]
            icd_label = display_label(icd_node, icd_data)
            icd_codes.append(icd_label)
            category = icd_data.get("code_category")
            if category:
                icd_categories.append(str(category))
            icd_records.append({"ICD-10": icd_label, "Disease": disease_label})
    rxcui_nodes = [dst for _, dst, attrs in graph.out_edges(node_id, data=True) if attrs.get("type") == "HAS_RXCUI"]
    precise: List[str] = []
    ingredients: List[str] = []
    brands: List[str] = []
    for rxcui_node in rxcui_nodes:
        for _, child, child_attrs in graph.out_edges(rxcui_node, data=True):
            rel_type = child_attrs.get("type")
            child_data = graph.nodes[child]
            label = display_label(child, child_data)
            if rel_type == "CONTAINS" and child_data.get("type") == "Ingredient":
                ingredients.append(label)
            if rel_type == "CONTAINS" and child_data.get("type") == "Precise_Ingredient":
                precise.append(label)
            if rel_type == "BRANDED_AS" and child_data.get("type") == "Brand":
                brands.append(label)
    unique_icd_pairs = {(item["ICD-10"], item["Disease"]): item for item in icd_records}
    icd_details = sorted(unique_icd_pairs.values(), key=lambda row: (row["ICD-10"], row["Disease"]))
    response = {
        "drug": display_label(node_id, data),
        "diseases": sorted(set(diseases)),
        "icd10_codes": sorted(set(icd_codes)),
        "icd10_root_codes": sorted(set(icd_categories)),
        "icd10_details": icd_details,
        "ingredients": sorted(set(ingredients or ensure_iterable(data.get("ingredients")))),
        "precise_ingredients": sorted(set(precise or ensure_iterable(data.get("precise_ingredients")))),
        "brand_names": sorted(set(brands or ensure_iterable(data.get("brand_names")))),
        "formulary": data.get("formulary", ""),
        "approval_date": approval_value,
        "dose_forms": ensure_iterable(data.get("dose_forms")),
        "rxcui": data.get("rxcui", ""),
    }
    return response


@st.cache_data(show_spinner=False)
def query_drug_diseases(graph_path: str, drug_name: str) -> Dict[str, Any]:
    graph = load_knowledge_graph(graph_path)
    return _query_drug(graph, drug_name)


def _drug_hierarchy(graph: nx.DiGraph, drug_name: str) -> Dict[str, Any]:
    node_id = find_node(graph, drug_name, "Drug")
    if not node_id:
        return {}
    data = graph.nodes[node_id]
    approval_raw = resolve_approval_value(data)
    approval_value = None
    if approval_raw:
        parsed = parse_iso_date(approval_raw)
        approval_value = parsed.isoformat() if parsed else approval_raw
    hierarchy: Dict[str, Any] = {
        "drug": display_label(node_id, data),
        "rxcui": data.get("rxcui"),
        "approval_date": approval_value,
        "diseases": [],
        "ingredients": [],
        "brand_names": [],
    }
    hierarchy["ingredients"] = ensure_iterable(data.get("ingredients"))
    hierarchy["brand_names"] = ensure_iterable(data.get("brand_names"))
    rxcui_nodes = [dst for _, dst, attrs in graph.out_edges(node_id, data=True) if attrs.get("type") == "HAS_RXCUI"]
    for disease_edge in graph.out_edges(node_id, data=True):
        if disease_edge[2].get("type") != "TREATS":
            continue
        disease_node = disease_edge[1]
        disease_data = graph.nodes[disease_node]
        disease_entry = {
            "name": display_label(disease_node, disease_data),
            "icd10_codes": [],
            "icd10_categories": [],
        }
        for _, icd_node, icd_attrs in graph.out_edges(disease_node, data=True):
            if icd_attrs.get("type") != "DIAGNOSED_AS":
                continue
            icd_data = graph.nodes[icd_node]
            icd_label = display_label(icd_node, icd_data)
            disease_entry["icd10_codes"].append(icd_label)
            category = icd_data.get("code_category")
            if category:
                disease_entry["icd10_categories"].append(str(category))
        hierarchy["diseases"].append(disease_entry)
    return hierarchy


@st.cache_data(show_spinner=False)
def get_drug_hierarchy(graph_path: str, drug_name: str) -> Dict[str, Any]:
    graph = load_knowledge_graph(graph_path)
    return _drug_hierarchy(graph, drug_name)


def _query_disease(graph: nx.DiGraph, disease_name: str) -> Dict[str, Any]:
    node_id = find_node(graph, disease_name, "Disease")
    if not node_id:
        return {}
    disease_data = graph.nodes[node_id]
    drugs: List[str] = []
    icd_mapping: Dict[str, str] = {}
    for src, _, attrs in graph.in_edges(node_id, data=True):
        if attrs.get("type") == "TREATS":
            drugs.append(display_label(src, graph.nodes[src]))
    for _, icd_node, attrs in graph.out_edges(node_id, data=True):
        if attrs.get("type") != "DIAGNOSED_AS":
            continue
        icd_data = graph.nodes[icd_node]
        icd_mapping[display_label(icd_node, icd_data)] = icd_data.get("code_category", "")
    return {
        "disease": display_label(node_id, disease_data),
        "treating_drugs": sorted(set(drugs)),
        "icd10_mapping": icd_mapping,
    }


@st.cache_data(show_spinner=False)
def query_disease_drugs(graph_path: str, disease_name: str) -> Dict[str, Any]:
    graph = load_knowledge_graph(graph_path)
    return _query_disease(graph, disease_name)


@st.cache_data(show_spinner=False)
def get_disease_icd10_mapping(graph_path: str, disease_name: str) -> Dict[str, str]:
    graph = load_knowledge_graph(graph_path)
    disease_info = _query_disease(graph, disease_name)
    return disease_info.get("icd10_mapping", {})


def _query_icd(graph: nx.DiGraph, icd_code: str) -> Dict[str, Any]:
    node_id = find_node(graph, icd_code, "ICD10Category")
    if not node_id:
        return {}
    node_data = graph.nodes[node_id]
    diseases: List[str] = []
    drugs: List[str] = []
    for src, _, attrs in graph.in_edges(node_id, data=True):
        if attrs.get("type") != "DIAGNOSED_AS":
            continue
        disease_data = graph.nodes[src]
        disease_label = display_label(src, disease_data)
        diseases.append(disease_label)
        for drug_src, _, drug_attrs in graph.in_edges(src, data=True):
            if drug_attrs.get("type") == "TREATS":
                drugs.append(display_label(drug_src, graph.nodes[drug_src]))
    return {
        "icd10_code": display_label(node_id, node_data),
        "description": node_data.get("code_category", ""),
        "diseases": sorted(set(diseases)),
        "drugs": sorted(set(drugs)),
    }


@st.cache_data(show_spinner=False)
def query_icd10_code(graph_path: str, icd_code: str) -> Dict[str, Any]:
    graph = load_knowledge_graph(graph_path)
    return _query_icd(graph, icd_code)


def _icd_hierarchy(graph: nx.DiGraph, icd_code: str) -> Dict[str, Any]:
    node_id = find_node(graph, icd_code, "ICD10Category")
    if not node_id:
        return {}
    node_data = graph.nodes[node_id]
    diseases: List[str] = []
    related_codes: List[str] = []
    for src, _, attrs in graph.in_edges(node_id, data=True):
        if attrs.get("type") == "DIAGNOSED_AS":
            diseases.append(display_label(src, graph.nodes[src]))
    prefix = re.match(r"^[A-Z]+", display_label(node_id, node_data))
    root = prefix.group(0) if prefix else display_label(node_id, node_data)[:1]
    for icd_node, data in graph.nodes(data=True):
        if data.get("type") != "ICD10Category":
            continue
        label = display_label(icd_node, data)
        if label.startswith(root) and icd_node != node_id:
            related_codes.append(label)
    return {
        "code": display_label(node_id, node_data),
        "chapter": node_data.get("code_category", ""),
        "diseases": sorted(set(diseases)),
        "related_codes": sorted(set(related_codes))[:50],
    }


@st.cache_data(show_spinner=False)
def get_icd10_hierarchy(graph_path: str, icd_code: str) -> Dict[str, Any]:
    graph = load_knowledge_graph(graph_path)
    return _icd_hierarchy(graph, icd_code)


def query_by_ingredient(graph: nx.DiGraph, ingredient: str) -> Dict[str, Any]:
    target = normalise_text(ingredient)
    result_drugs: List[str] = []
    diseases: List[str] = []
    for node, data in graph.nodes(data=True):
        if data.get("type") != "Drug":
            continue
        ingredients = [normalise_text(item) for item in ensure_iterable(data.get("ingredients"))]
        if target and not any(target in ing for ing in ingredients):
            continue
        result_drugs.append(display_label(node, data))
        for _, disease_node, attrs in graph.out_edges(node, data=True):
            if attrs.get("type") == "TREATS":
                diseases.append(display_label(disease_node, graph.nodes[disease_node]))
    return {
        "ingredient": ingredient,
        "drugs": sorted(set(result_drugs)),
        "diseases": sorted(set(diseases)),
    }


def query_by_approval_date(graph: nx.DiGraph, start: Optional[date], end: Optional[date]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for node, data in graph.nodes(data=True):
        if data.get("type") != "Drug":
            continue
        approval_raw = resolve_approval_value(data)
        approval_date = parse_iso_date(approval_raw)
        if start and (not approval_date or approval_date < start):
            continue
        if end and (not approval_date or approval_date > end):
            continue
        results.append(
            {
                "drug": display_label(node, data),
                "approval_date": approval_date.isoformat() if approval_date else (approval_raw or "Unknown"),
                "rxcui": data.get("rxcui", ""),
            }
        )
    results.sort(key=lambda item: item.get("approval_date", ""))
    return results


def advanced_query(
    graph: nx.DiGraph,
    drug_name: Optional[str] = None,
    disease_name: Optional[str] = None,
    icd_code: Optional[str] = None,
    ingredient: Optional[str] = None,
    approval_start: Optional[date] = None,
    approval_end: Optional[date] = None,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for node, data in graph.nodes(data=True):
        if data.get("type") != "Drug":
            continue
        label = display_label(node, data)
        if drug_name and normalise_text(drug_name) not in normalise_text(label):
            continue
        approval_raw = resolve_approval_value(data)
        approval_date = parse_iso_date(approval_raw)
        if approval_start and (not approval_date or approval_date < approval_start):
            continue
        if approval_end and (not approval_date or approval_date > approval_end):
            continue
        diseases = [display_label(dst, graph.nodes[dst]) for _, dst, attrs in graph.out_edges(node, data=True) if attrs.get("type") == "TREATS"]
        if disease_name and all(normalise_text(disease_name) not in normalise_text(d) for d in diseases):
            continue
        icd_codes: List[str] = []
        for _, disease_node, attrs in graph.out_edges(node, data=True):
            if attrs.get("type") != "TREATS":
                continue
            for _, icd_node, icd_attrs in graph.out_edges(disease_node, data=True):
                if icd_attrs.get("type") == "DIAGNOSED_AS":
                    icd_codes.append(display_label(icd_node, graph.nodes[icd_node]))
        if icd_code and all(normalise_text(icd_code) not in normalise_text(code) for code in icd_codes):
            continue
        ing_list = ensure_iterable(data.get("ingredients"))
        if ingredient and all(normalise_text(ingredient) not in normalise_text(ing) for ing in ing_list):
            continue
        results.append(
            {
                "drug": label,
                "diseases": sorted(set(diseases)),
                "icd10_codes": sorted(set(icd_codes)),
                "ingredients": ing_list,
                "approval_date": approval_date.isoformat() if approval_date else (approval_raw or "Unknown"),
            }
        )
    return results


# ---------------------------------------------------------------------------
# Validation and suggestion helpers
# ---------------------------------------------------------------------------


def validate_drug_input(graph_path: str, drug_name: str) -> Tuple[bool, str]:
    if not drug_name:
        return False, "Please provide a drug name."
    graph = load_knowledge_graph(graph_path)
    if find_node(graph, drug_name, "Drug"):
        return True, ""
    suggestions = fuzzy_search_candidates(graph_path, drug_name, "Drug")
    if suggestions:
        return False, f"Drug not found. Did you mean: {', '.join(suggestions)}?"
    return False, "Drug not found in the knowledge graph."


def validate_icd10_code(graph_path: str, code: str) -> Tuple[bool, str]:
    if not code:
        return False, "Enter an ICD-10 code."
    graph = load_knowledge_graph(graph_path)
    if find_node(graph, code, "ICD10Category"):
        return True, ""
    if not ICD_CODE_PATTERN.match(code.upper()):
        return False, "Invalid ICD-10 format. Use values like A10.1 or B20."
    suggestions = fuzzy_search_candidates(graph_path, code, "ICD10Category")
    if suggestions:
        return False, f"Code not found. Suggestions: {', '.join(suggestions)}."
    return False, "ICD-10 code not present in the graph."


def fuzzy_search_suggestions(graph_path: str, query: str, search_type: str) -> List[str]:
    return fuzzy_search_candidates(graph_path, query, search_type)


def show_helpful_error(error_type: str, message: str) -> None:
    st.warning(f"{error_type.capitalize()} lookup issue: {message}")


def show_search_tips() -> None:
    st.info(
        "Try entering partial terms (e.g. 'metform' or 'A10') to surface autocomplete suggestions."
    )


def show_data_coverage_info() -> None:
    st.caption(
        "Data sources include RxNav drug metadata, FDA approval dates and ICD-10 mappings generated by the pipeline."
    )


# ---------------------------------------------------------------------------
# Visualisation builders
# ---------------------------------------------------------------------------


def create_drug_network_viz(graph: nx.DiGraph, drug_name: str) -> go.Figure:
    node_id = find_node(graph, drug_name, "Drug")
    if not node_id:
        return go.Figure()
    sub_nodes = {node_id}
    for _, disease_node, attrs in graph.out_edges(node_id, data=True):
        if attrs.get("type") == "TREATS":
            sub_nodes.add(disease_node)
            for _, icd_node, icd_attrs in graph.out_edges(disease_node, data=True):
                if icd_attrs.get("type") == "DIAGNOSED_AS":
                    sub_nodes.add(icd_node)
    rxcui_nodes = [dst for _, dst, attrs in graph.out_edges(node_id, data=True) if attrs.get("type") == "HAS_RXCUI"]
    for rxcui_node in rxcui_nodes:
        sub_nodes.add(rxcui_node)
        for _, child, _ in graph.out_edges(rxcui_node, data=True):
            sub_nodes.add(child)
    subgraph = graph.subgraph(sub_nodes).copy()
    if subgraph.number_of_nodes() > 150:
        subgraph = subgraph
    pos = nx.spring_layout(subgraph, seed=42, k=0.6)
    node_types = {}
    for node, data in subgraph.nodes(data=True):
        node_types.setdefault(data.get("type", "Other"), []).append(node)
    figures = []
    for node_type, nodes in node_types.items():
        x_vals = [pos[node][0] for node in nodes]
        y_vals = [pos[node][1] for node in nodes]
        labels = [display_label(node, subgraph.nodes[node]) for node in nodes]
        figures.append(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="markers",
                name=node_type,
                marker=dict(size=16 if node_type == "Drug" else 10),
                text=labels,
                hovertext=labels,
                hoverinfo="text",
            )
        )
    edge_x: List[float] = []
    edge_y: List[float] = []
    for src, dst in subgraph.edges():
        x0, y0 = pos[src]
        x1, y1 = pos[dst]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=1, color="#BBBBBB"), hoverinfo="none", showlegend=False)
    fig = go.Figure()
    fig.add_trace(edge_trace)
    for trace in figures:
        fig.add_trace(trace)
    fig.update_layout(
        title=f"Knowledge graph neighbourhood for {drug_name}",
        showlegend=True,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def create_disease_hierarchy_viz(disease_data: Dict[str, Any]) -> go.Figure:
    icd_codes = disease_data.get("icd10_mapping", {})
    if not icd_codes:
        return go.Figure()
    labels = []
    parents = []
    values = []
    for code, description in icd_codes.items():
        labels.append(code)
        parents.append(description or "ICD-10")
        values.append(1)
    fig = px.treemap(
        path=[parents, labels],
        values=values,
        title=f"ICD-10 hierarchy for {disease_data.get('disease', 'Disease')}",
    )
    return fig


def create_approval_timeline_viz(drugs_data: List[Dict[str, Any]]) -> go.Figure:
    if not drugs_data:
        return go.Figure()
    timeline = {}
    for record in drugs_data:
        approval = record.get("approval_date")
        if not approval or approval == "Unknown":
            continue
        year = approval[:4]
        timeline[year] = timeline.get(year, 0) + 1
    if not timeline:
        return go.Figure()
    years = sorted(timeline)
    counts = [timeline[year] for year in years]
    fig = go.Figure(go.Bar(x=years, y=counts))
    fig.update_layout(title="Drug approvals per year", xaxis_title="Year", yaxis_title="Approvals")
    return fig


def create_ingredient_analysis_viz(ingredient_data: Dict[str, Any]) -> go.Figure:
    drugs = ingredient_data.get("drugs", [])
    if not drugs:
        return go.Figure()
    df = pd.DataFrame({"drugs": drugs})
    df["count"] = 1
    df = df.groupby("drugs").sum().reset_index().sort_values("count", ascending=False)
    fig = px.bar(df.head(30), x="drugs", y="count", title="Drugs containing the ingredient", labels={"drugs": "Drug", "count": "Count"})
    fig.update_layout(xaxis_tickangle=-45)
    return fig


# ---------------------------------------------------------------------------
# UI helper components
# ---------------------------------------------------------------------------


def load_custom_css() -> None:
    st.markdown(
        """
        <style>
        .main > div { padding-top: 2rem; }
        .stSelectbox div[data-baseweb="select"] {
            background-color: #1f2933 !important;
            color: #e2e8f0 !important;
            border-radius: 10px;
        }
        .stSelectbox div[data-baseweb="select"] > div {
            color: #e2e8f0 !important;
        }
        .stSelectbox div[data-baseweb="select"] svg {
            fill: #e2e8f0 !important;
        }
        .stSelectbox div[data-baseweb="popover"] li,
        .stSelectbox div[data-baseweb="popover"] li > div {
            background-color: #1f2933 !important;
            color: #e2e8f0 !important;
        }
        .stTextInput > div > div,
        .stTextInput > div > div > input {
            background-color: #1f2933 !important;
            color: #e2e8f0 !important;
        }
        .stTextInput > div > div > input::placeholder {
            color: #94a3b8 !important;
        }
        .metric-card { border: 1px solid #e6e9ef; border-radius: 8px; padding: 1rem; margin-bottom: 1rem; }
        .metric-card h3 { margin: 0 0 0.5rem 0; font-size: 1.1rem; }
        .metric-card p { margin: 0; font-size: 0.9rem; color: #4b4b4b; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def create_drug_summary_card(drug_data: Dict[str, Any]) -> None:
    if not drug_data:
        st.info("Select a drug to view details.")
        return
    cols = st.columns(3)
    cols[0].metric("RxCUI", drug_data.get("rxcui", "-"))
    cols[1].metric("Diseases", len(drug_data.get("diseases", [])))
    cols[2].metric("ICD-10 codes", len(drug_data.get("icd10_codes", [])))
    st.markdown("---")
    st.subheader(f"Drug overview – {drug_data.get('drug', '')}")
    overview_rows = [
        {"Attribute": "Approval date", "Value": drug_data.get("approval_date") or "Unknown"},
        {"Attribute": "Brand names", "Value": ", ".join(drug_data.get("brand_names", [])) or "None"},
        {"Attribute": "Ingredients", "Value": ", ".join(drug_data.get("ingredients", [])) or "None"},
        {"Attribute": "Precise ingredients", "Value": ", ".join(drug_data.get("precise_ingredients", [])) or "None"},
    ]
    if drug_data.get("formulary"):
        overview_rows.append({"Attribute": "Formulary", "Value": drug_data.get("formulary")})
    st.table(pd.DataFrame(overview_rows))


def create_metrics_dashboard(graph: nx.DiGraph) -> None:
    col1, col2, col3 = st.columns(3)
    col1.metric("Drugs", sum(1 for _, data in graph.nodes(data=True) if data.get("type") == "Drug"))
    col2.metric("Diseases", sum(1 for _, data in graph.nodes(data=True) if data.get("type") == "Disease"))
    col3.metric("ICD-10 codes", sum(1 for _, data in graph.nodes(data=True) if data.get("type") == "ICD10Category"))
    st.markdown("---")
    degree_df = build_degree_dataframe(st.session_state.graph_path)
    top_nodes = degree_df.sort_values(["out_degree"], ascending=False).head(10)
    st.subheader("Most connected nodes")
    st.dataframe(top_nodes)


def create_comparison_table(items_data: List[Dict[str, Any]]) -> None:
    if not items_data:
        st.info("No results to compare.")
        return
    df = pd.DataFrame(items_data)
    st.dataframe(df)


def create_drug_search_interface(graph_path: str) -> Optional[str]:
    drugs = get_all_drugs(graph_path)
    selected = st.selectbox("Select a drug", options=["-- Choose --"] + drugs)
    manual_input = st.text_input("Or search by name", value="")
    target = manual_input or (selected if selected != "-- Choose --" else "")
    if not target:
        show_search_tips()
        return None
    valid, message = validate_drug_input(graph_path, target)
    if not valid:
        show_helpful_error("drug", message)
        suggestions = fuzzy_search_suggestions(graph_path, target, "Drug")
        if suggestions:
            st.write("Suggestions:", ", ".join(suggestions))
        return None
    return target


def create_disease_search_interface(graph_path: str) -> Optional[str]:
    diseases = get_all_diseases(graph_path)
    selected = st.selectbox("Select a disease", options=["-- Choose --"] + diseases)
    manual_input = st.text_input("Or search by disease name", value="")
    target = manual_input or (selected if selected != "-- Choose --" else "")
    if not target:
        show_search_tips()
        return None
    graph = load_knowledge_graph(graph_path)
    if not find_node(graph, target, "Disease"):
        show_helpful_error("disease", "No match found.")
        suggestions = fuzzy_search_suggestions(graph_path, target, "Disease")
        if suggestions:
            st.write("Suggestions:", ", ".join(suggestions))
        return None
    return target


def create_icd10_search_interface(graph_path: str) -> Optional[str]:
    codes = get_all_icd10_codes(graph_path)
    selected = st.selectbox("Select an ICD-10 code", options=["-- Choose --"] + codes)
    manual_input = st.text_input("Or search by code", value="")
    target = manual_input or (selected if selected != "-- Choose --" else "")
    if not target:
        show_search_tips()
        return None
    valid, message = validate_icd10_code(graph_path, target)
    if not valid:
        show_helpful_error("icd-10", message)
        return None
    return target


def create_advanced_search_interface() -> Dict[str, Any]:
    with st.form("advanced_search"):
        drug_filter = st.text_input("Drug name contains")
        disease_filter = st.text_input("Disease name contains")
        icd_filter = st.text_input("ICD-10 code contains")
        ingredient_filter = st.text_input("Ingredient contains")
        col1, col2 = st.columns(2)
        with col1:
            use_start = st.checkbox("Filter by start date", value=False)
            start_date = st.date_input("Approval start", value=date.today(), disabled=not use_start)
        with col2:
            use_end = st.checkbox("Filter by end date", value=False)
            end_date = st.date_input("Approval end", value=date.today(), disabled=not use_end)
        submitted = st.form_submit_button("Search")
    return {
        "submitted": submitted,
        "drug_filter": drug_filter or None,
        "disease_filter": disease_filter or None,
        "icd_filter": icd_filter or None,
        "ingredient_filter": ingredient_filter or None,
        "start_date": start_date if "use_start" in locals() and use_start else None,
        "end_date": end_date if "use_end" in locals() and use_end else None,
    }


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------


def export_to_excel(data: Dict[str, Any], filename: str) -> Tuple[str, bytes]:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for key, value in data.items():
            if isinstance(value, list):
                pd.DataFrame({key: value}).to_excel(writer, sheet_name=key[:31] or "Sheet1", index=False)
            elif isinstance(value, dict):
                pd.DataFrame(list(value.items()), columns=["Key", "Value"]).to_excel(writer, sheet_name=key[:31] or "Sheet1", index=False)
            else:
                pd.DataFrame({key: [value]}).to_excel(writer, sheet_name=key[:31] or "Sheet1", index=False)
    return filename, buffer.getvalue()


def export_to_csv(data: Dict[str, Any], filename: str) -> Tuple[str, bytes]:
    buffer = BytesIO()
    flattened: Dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, (list, tuple)):
            flattened[key] = "; ".join(str(item) for item in value)
        elif isinstance(value, dict):
            flattened[key] = json.dumps(value)
        else:
            flattened[key] = value
    pd.DataFrame([flattened]).to_csv(buffer, index=False)
    return filename, buffer.getvalue()


def create_pdf_report(query_results: Dict[str, Any]) -> Tuple[Optional[str], Optional[bytes]]:
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
    except ImportError:
        return None, None
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    pdf.setTitle("Drug Knowledge Graph Report")
    y = height - 50
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(50, y, "Drug Knowledge Graph Report")
    y -= 30
    pdf.setFont("Helvetica", 10)
    for key, value in query_results.items():
        if y < 80:
            pdf.showPage()
            y = height - 50
        pdf.drawString(50, y, f"{key}:")
        y -= 15
        text = json.dumps(value, indent=2) if isinstance(value, (dict, list)) else str(value)
        for line in text.splitlines():
            if y < 80:
                pdf.showPage()
                y = height - 50
            pdf.drawString(60, y, line[:100])
            y -= 12
        y -= 10
    pdf.save()
    return "knowledge_graph_report.pdf", buffer.getvalue()


def export_graph_data(subgraph: nx.DiGraph, filename: str = "graph_export.zip") -> Tuple[str, bytes]:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        temp_graphml = nx.generate_graphml(subgraph)
        archive.writestr("subgraph.graphml", "".join(temp_graphml))
        node_rows = []
        for node, data in subgraph.nodes(data=True):
            row = {"id": node}
            row.update(data)
            node_rows.append(row)
        edge_rows = []
        for src, dst, attrs in subgraph.edges(data=True):
            row = {"source": src, "target": dst}
            row.update(attrs)
            edge_rows.append(row)
        archive.writestr("nodes.csv", pd.DataFrame(node_rows).to_csv(index=False))
        archive.writestr("edges.csv", pd.DataFrame(edge_rows).to_csv(index=False))
    return filename, buffer.getvalue()


# ---------------------------------------------------------------------------
# Sidebar and session-state management
# ---------------------------------------------------------------------------


def initialise_session_state() -> None:
    if "graph_path" not in st.session_state:
        default_path = find_default_graph_path()
        st.session_state.graph_path = default_path or ""
    if "query_history" not in st.session_state:
        st.session_state.query_history: List[Dict[str, Any]] = []
    if "user_preferences" not in st.session_state:
        st.session_state.user_preferences = {"show_tips": True}
    if "selected_page" not in st.session_state:
        st.session_state.selected_page = PAGE_NAMES[0]


def append_history(entry: Dict[str, Any]) -> None:
    st.session_state.query_history.insert(0, entry)
    if len(st.session_state.query_history) > MAX_HISTORY_ENTRIES:
        st.session_state.query_history = st.session_state.query_history[:MAX_HISTORY_ENTRIES]


def create_sidebar(graph_loaded: bool) -> str:
    st.sidebar.title("🔍 Navigation")
    graph_input = st.sidebar.text_input("Knowledge graph path", st.session_state.graph_path or "")
    if graph_input and graph_input != st.session_state.graph_path:
        st.session_state.graph_path = graph_input
        st.cache_data.clear()
        st.experimental_rerun()
    if not graph_loaded:
        st.sidebar.warning("Load a valid knowledge graph to enable features.")
    selected = st.sidebar.radio("Select page", PAGE_NAMES, index=PAGE_NAMES.index(st.session_state.selected_page))
    st.session_state.selected_page = selected
    st.sidebar.markdown("---")
    quick_search = st.sidebar.text_input("Quick search (drug/disease/ICD)")
    if quick_search and graph_loaded:
        graph = load_knowledge_graph(st.session_state.graph_path)
        for node_type in ("Drug", "Disease", "ICD10Category"):
            node_id = find_node(graph, quick_search, node_type)
            if node_id:
                append_history({"type": node_type, "query": quick_search, "timestamp": datetime.utcnow().isoformat(), "status": "Quick search"})
                if node_type == "Drug":
                    st.session_state.selected_page = "Drug Lookup"
                elif node_type == "Disease":
                    st.session_state.selected_page = "Disease Explorer"
                else:
                    st.session_state.selected_page = "ICD-10 Navigator"
                st.experimental_rerun()
    with st.sidebar.expander("Query history", expanded=False):
        if st.session_state.query_history:
            for entry in st.session_state.query_history[:10]:
                st.write(f"{entry.get('timestamp', '')}: {entry.get('type')} – {entry.get('query')}")
            if st.button("Clear history", key="clear_history"):
                st.session_state.query_history.clear()
        else:
            st.write("No queries yet.")
    return selected


# ---------------------------------------------------------------------------
# Page implementations
# ---------------------------------------------------------------------------


def home_page(graph: nx.DiGraph) -> None:
    st.title("💊 Drug Knowledge Graph Explorer")
    st.write("Explore pharmaceutical relationships between drugs, diseases, ingredients, and ICD-10 codes.")
    show_data_coverage_info()
    create_metrics_dashboard(graph)
    degree_df = build_degree_dataframe(st.session_state.graph_path)
    type_counts = degree_df.groupby("type").size().reset_index(name="count")
    fig = px.pie(type_counts, names="type", values="count", title="Node type distribution")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("### Recent query history")
    if st.session_state.query_history:
        st.dataframe(pd.DataFrame(st.session_state.query_history))
    else:
        st.info("No queries executed yet.")


def drug_lookup_page(graph_path: str, graph: nx.DiGraph) -> None:
    st.title("💊 Drug Information Lookup")
    target = create_drug_search_interface(graph_path)
    if target:
        result = query_drug_diseases(graph_path, target)
        create_drug_summary_card(result)
        append_history({"type": "Drug", "query": target, "timestamp": datetime.utcnow().isoformat(), "status": f"Diseases: {len(result.get('diseases', []))}"})
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("Diseases treated")
            st.dataframe(pd.DataFrame({"Diseases": result.get("diseases", [])}))
            st.subheader("ICD-10 codes")
            icd_details = result.get("icd10_details", []) or []
            if icd_details:
                icd_df = pd.DataFrame(icd_details)[["Disease", "ICD-10"]]
            else:
                icd_df = pd.DataFrame(columns=["Disease", "ICD-10"])
            st.dataframe(icd_df.sort_values(by = "Disease"))
        with col2:
            st.subheader("Ingredients")
            st.dataframe(pd.DataFrame({"Ingredient": result.get("ingredients", [])}))
            st.subheader("Brand names")
            st.dataframe(pd.DataFrame({"Brand": result.get("brand_names", [])}))
        st.markdown("### Exports")
        excel_name, excel_bytes = export_to_excel(result, "drug_query.xlsx")
        st.download_button("Download Excel", data=excel_bytes, file_name=excel_name)
        csv_name, csv_bytes = export_to_csv(result, "drug_query.csv")
        st.download_button("Download CSV", data=csv_bytes, file_name=csv_name)
        pdf_name, pdf_bytes = create_pdf_report(result)
        if pdf_name and pdf_bytes:
            st.download_button("Download PDF", data=pdf_bytes, file_name=pdf_name)
        else:
            st.info("Install `reportlab` to enable PDF export.")
        st.markdown("### Network view")
        fig = create_drug_network_viz(graph, target)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Select or search for a drug to begin.")


def disease_explorer_page(graph_path: str) -> None:
    st.title("🩺 Disease Explorer")
    target = create_disease_search_interface(graph_path)
    if target:
        result = query_disease_drugs(graph_path, target)
        append_history({"type": "Disease", "query": target, "timestamp": datetime.utcnow().isoformat(), "status": f"Drugs: {len(result.get('treating_drugs', []))}"})
        st.subheader(f"Drugs treating {result.get('disease')}")
        st.dataframe(pd.DataFrame({"Drugs": result.get("treating_drugs", [])}))
        st.subheader("ICD-10 mapping")
        mapping = result.get("icd10_mapping", {})
        if mapping:
            mapping_df = pd.DataFrame({"ICD-10": list(mapping.keys()), "Description": list(mapping.values())})
            st.dataframe(mapping_df)
            viz = create_disease_hierarchy_viz(result)
            st.plotly_chart(viz, use_container_width=True)
        else:
            st.info("No ICD-10 codes associated.")
        st.markdown("### Export results")
        excel_name, excel_bytes = export_to_excel(result, "disease_query.xlsx")
        st.download_button("Download Excel", data=excel_bytes, file_name=excel_name)
    else:
        st.info("Select or search for a disease.")


def icd10_navigator_page(graph_path: str) -> None:
    st.title("🧬 ICD-10 Navigator")
    target = create_icd10_search_interface(graph_path)
    if target:
        result = query_icd10_code(graph_path, target)
        append_history({"type": "ICD-10", "query": target, "timestamp": datetime.utcnow().isoformat(), "status": f"Diseases: {len(result.get('diseases', []))}"})
        st.subheader(f"Diseases linked to {result.get('icd10_code')}")
        st.dataframe(pd.DataFrame({"Diseases": result.get("diseases", [])}))
        st.subheader("Drugs")
        st.dataframe(pd.DataFrame({"Drugs": result.get("drugs", [])}))
        hierarchy = get_icd10_hierarchy(graph_path, target)
        st.write(hierarchy)
    else:
        st.info("Select or search for an ICD-10 code.")


def advanced_search_page(graph: nx.DiGraph) -> None:
    st.title("🔬 Advanced Search")
    filters = create_advanced_search_interface()
    if filters["submitted"]:
        results = advanced_query(
            graph,
            drug_name=filters["drug_filter"],
            disease_name=filters["disease_filter"],
            icd_code=filters["icd_filter"],
            ingredient=filters["ingredient_filter"],
            approval_start=filters["start_date"],
            approval_end=filters["end_date"],
        )
        append_history({"type": "Advanced", "query": json.dumps({
            "drug": filters["drug_filter"],
            "disease": filters["disease_filter"],
            "icd": filters["icd_filter"],
            "ingredient": filters["ingredient_filter"],
        }), "timestamp": datetime.utcnow().isoformat(), "status": f"Results: {len(results)}"})
        if results:
            st.success(f"Found {len(results)} matching drugs.")
            create_comparison_table(results)
            timeline_fig = create_approval_timeline_viz(results)
            st.plotly_chart(timeline_fig, use_container_width=True)
        else:
            st.warning("No matches. Adjust your filters and try again.")
    st.markdown("---")
    st.subheader("Ingredient quick lookup")
    ingredient_query = st.text_input("Ingredient based lookup", key="ingredient_lookup")
    if ingredient_query:
        ingredient_result = query_by_ingredient(graph, ingredient_query)
        if ingredient_result.get("drugs"):
            st.dataframe(pd.DataFrame({"Drugs": ingredient_result.get("drugs", [])}))
            st.dataframe(pd.DataFrame({"Diseases": ingredient_result.get("diseases", [])}))
        else:
            show_helpful_error("ingredient", "No drugs use this ingredient.")
    st.markdown("---")
    st.subheader("Approval date range lookup")
    col1, col2 = st.columns(2)
    with col1:
        use_start_range = st.checkbox("Use start date", key="approval_use_start", value=False)
        start = st.date_input("Start date", key="approval_start", value=date.today(), disabled=not use_start_range)
    with col2:
        use_end_range = st.checkbox("Use end date", key="approval_use_end", value=False)
        end = st.date_input("End date", key="approval_end", value=date.today(), disabled=not use_end_range)
    if st.button("Find approvals"):
        results = query_by_approval_date(
            graph,
            start if use_start_range else None,
            end if use_end_range else None,
        )
        if results:
            st.dataframe(pd.DataFrame(results))
        else:
            show_helpful_error("approval", "No approvals in the selected range.")


def visualization_page(graph_path: str, graph: nx.DiGraph) -> None:
    st.title("📊 Knowledge Graph Visualization")
    options = ["Drug neighbourhood", "Degree distribution", "Ingredient analysis"]
    viz_type = st.selectbox("Choose visualization type", options)
    if viz_type == "Drug neighbourhood":
        drug = st.selectbox("Select drug", ["-- Choose --"] + get_all_drugs(graph_path))
        if drug != "-- Choose --":
            fig = create_drug_network_viz(graph, drug)
            st.plotly_chart(fig, use_container_width=True)
    elif viz_type == "Degree distribution":
        df = build_degree_dataframe(graph_path)
        fig = px.scatter(df, x="in_degree", y="out_degree", color="type", hover_name="node", title="Node degree distribution")
        st.plotly_chart(fig, use_container_width=True)
    elif viz_type == "Ingredient analysis":
        ingredient = st.selectbox("Ingredient", ["-- Choose --"] + sorted({ing for _, data in graph.nodes(data=True) if data.get("type") == "Drug" for ing in ensure_iterable(data.get("ingredients"))}))
        if ingredient != "-- Choose --":
            data = query_by_ingredient(graph, ingredient)
            fig = create_ingredient_analysis_viz(data)
            st.plotly_chart(fig, use_container_width=True)


def data_export_page(graph: nx.DiGraph) -> None:
    st.title("📁 Data Export")
    st.write("Download slices of the knowledge graph for offline analysis.")
    show_data_coverage_info()
    drug = st.selectbox("Select drug for export", ["-- Choose --"] + get_all_drugs(st.session_state.graph_path))
    if drug != "-- Choose --":
        node_id = find_node(graph, drug, "Drug")
        if node_id:
            neighbours = {node_id}
            for edge in graph.out_edges(node_id):
                neighbours.add(edge[1])
            subgraph = graph.subgraph(neighbours).copy()
            file_name, payload = export_graph_data(subgraph)
            st.download_button("Download subgraph (ZIP)", data=payload, file_name=file_name)
    st.markdown("---")
    st.write("Export entire graph metadata")
    nodes = []
    for node, data in graph.nodes(data=True):
        row = {"id": node}
        row.update(data)
        nodes.append(row)
    edges = []
    for src, dst, attrs in graph.edges(data=True):
        row = {"source": src, "target": dst}
        row.update(attrs)
        edges.append(row)
    st.download_button("Download nodes CSV", data=pd.DataFrame(nodes).to_csv(index=False), file_name="nodes.csv")
    st.download_button("Download edges CSV", data=pd.DataFrame(edges).to_csv(index=False), file_name="edges.csv")


def analytics_dashboard_page(graph_path: str, graph: nx.DiGraph) -> None:
    st.title("📈 Analytics Dashboard")
    degree_df = build_degree_dataframe(graph_path)
    st.subheader("Top 15 drugs by treated diseases")
    drug_counts = []
    for node, data in graph.nodes(data=True):
        if data.get("type") != "Drug":
            continue
        count = sum(1 for _, _, attrs in graph.out_edges(node, data=True) if attrs.get("type") == "TREATS")
        drug_counts.append({"drug": display_label(node, data), "diseases": count})
    df_drugs = pd.DataFrame(drug_counts).sort_values("diseases", ascending=False).head(15)
    st.bar_chart(df_drugs.set_index("drug"))
    st.subheader("Most common ingredients")
    ingredient_counter: Dict[str, int] = {}
    for node, data in graph.nodes(data=True):
        if data.get("type") == "Drug":
            for ing in ensure_iterable(data.get("ingredients")):
                ingredient_counter[ing] = ingredient_counter.get(ing, 0) + 1
    ingredient_df = pd.DataFrame(sorted(ingredient_counter.items(), key=lambda item: item[1], reverse=True), columns=["Ingredient", "Count"]).head(20)
    st.dataframe(ingredient_df)
    st.subheader("ICD-10 coverage by chapter")
    icd_df = degree_df[degree_df["type"] == "ICD10Category"]
    chapter_counter = icd_df["node"].apply(lambda code: code[:1]).value_counts().reset_index()
    chapter_counter.columns = ["Chapter", "Codes"]
    st.plotly_chart(px.bar(chapter_counter, x="Chapter", y="Codes", title="ICD-10 chapter coverage"), use_container_width=True)


# ---------------------------------------------------------------------------
# Application entry point
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(
        page_title="Drug Knowledge Graph Explorer",
        page_icon="💊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    load_custom_css()
    initialise_session_state()
    graph_loaded = False
    graph = None
    if st.session_state.graph_path:
        try:
            graph = load_knowledge_graph(st.session_state.graph_path)
            graph_loaded = True
        except Exception as exc:  # pylint: disable=broad-except
            st.error(f"Failed to load knowledge graph: {exc}")
    selected_page = create_sidebar(graph_loaded)
    if not graph_loaded or graph is None:
        st.warning("Provide a valid graph file to start exploring.")
        return
    if selected_page == "Home":
        home_page(graph)
    elif selected_page == "Drug Lookup":
        drug_lookup_page(st.session_state.graph_path, graph)
    elif selected_page == "Disease Explorer":
        disease_explorer_page(st.session_state.graph_path)
    elif selected_page == "ICD-10 Navigator":
        icd10_navigator_page(st.session_state.graph_path)
    elif selected_page == "Advanced Search":
        advanced_search_page(graph)
    elif selected_page == "Graph Visualization":
        visualization_page(st.session_state.graph_path, graph)
    elif selected_page == "Data Export":
        data_export_page(graph)
    elif selected_page == "Analytics Dashboard":
        analytics_dashboard_page(st.session_state.graph_path, graph)


if __name__ == "__main__":
    main()
