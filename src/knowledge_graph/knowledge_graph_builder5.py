#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pharma_kg_all.py

Build a SINGLE knowledge graph/tree from the entire Excel file (no per-drug
filtering). The graph includes ICD-10 hierarchy (root → full code → disease),
all drugs that treat those diseases, approval dates, RxCUI and RxCUI metadata
(brand, dose forms, ingredients, precise ingredients, formulary).

Outputs:
  - GraphML (.graphml), JSON (.json), Pickle (.pkl)
  - Neo4j CSVs (nodes, relationships)
  - Adjacency matrices (ICD x Disease, Disease x Drug)
  - Static PNG snapshot (optional radial snapshot around top drugs)
  - Interactive HTML (pyvis) if available

Requirements:
  pandas, networkx, matplotlib, openpyxl
  (optional) pyvis for interactive HTML

Usage (ALL drugs in one graph):
  python src/knowledge_graph/knowledge_graph_builder5.py --excel data/output/rxnav_with_fda_with_orig.xlsx --outdir data/output/kg"""

import argparse
import ast
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple
# from networkx.readwrite.gpickle import write_gpickle
import pickle

import pandas as pd
import matplotlib.pyplot as plt

try:
    import networkx as nx
except ImportError as e:
    raise SystemExit("ERROR: install networkx (pip install networkx).") from e

try:
    from pyvis.network import Network
    _PYVIS = True
except Exception:
    _PYVIS = False

ICD10_PATTERN = re.compile(r"^[A-TV-Z][0-9]{2}(?:\.[0-9A-TV-Z]{1,4})?$")

REQUIRED_COLUMNS = [
    "Drug Name", "Formulary", "RxCUI", "Brand Name",
    "Branded Dose Form Group", "Branded Drug Component", "Branded Drug or Pack",
    "Clinical Dose Form Group", "Clinical Drug Component", "Clinical Drug or Pack",
    "Dose Form Group", "Ingredient", "Precise Ingredient",
    "may_treat_diseases", "may_treat_diseases_icd10",
    "orig_approval_date",
]

# ---------- helpers ----------

def _clean_str(x) -> Optional[str]:
    if pd.isna(x):
        return None
    s = str(x).strip()
    return s if s else None

def _parse_list_cell(cell) -> List[str]:
    """Parse list-like cells that might be literal lists or delimited strings."""
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return []
    if isinstance(cell, list):
        return [str(i).strip() for i in cell if str(i).strip()]
    if isinstance(cell, str):
        # Try Python literal list
        try:
            v = ast.literal_eval(cell)
            if isinstance(v, list):
                return [str(i).strip() for i in v if str(i).strip()]
        except Exception:
            pass
        # Fallback delimiters
        return [s.strip() for s in re.split(r"[;,]", cell) if s.strip()]
    return [str(cell).strip()]

def _parse_icd_mapping(cell) -> Dict[str, List[str]]:
    """
    `may_treat_diseases_icd10` expected like:
        "[{'Pain': ['R52']}, {'Heroin Dependence': ['F11.20','F11.21']}]"
    Return a dict { disease: [codes...] }
    """
    out: Dict[str, List[str]] = {}
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return out
    try:
        v = ast.literal_eval(cell) if isinstance(cell, str) else cell
        if isinstance(v, list):
            for d in v:
                if isinstance(d, dict):
                    for disease, codes in d.items():
                        dz = _clean_str(disease)
                        if not dz:
                            continue
                        clist = []
                        if isinstance(codes, list):
                            for c in codes:
                                cc = _clean_str(c)
                                if cc:
                                    clist.append(cc.upper())
                        elif isinstance(codes, str):
                            clist.extend([s.strip().upper() for s in re.split(r"[;,]", codes) if s.strip()])
                        if clist:
                            out.setdefault(dz, []).extend(clist)
    except Exception:
        return {}
    return out

def _icd_root(full_code: str) -> Optional[str]:
    return full_code.split(".")[0][:3].upper() if full_code else None

def _add_node(G: nx.DiGraph, nid: str, **attrs):
    if nid not in G:
        G.add_node(nid)
    for k, v in attrs.items():
        if v is not None:
            G.nodes[nid][k] = v

def _add_edge(G: nx.DiGraph, u: str, v: str, **attrs):
    if not G.has_edge(u, v):
        G.add_edge(u, v, **attrs)
    else:
        G.edges[u, v].update({k: v for k, v in attrs.items() if v is not None})

# ---------- pipeline ----------

def load_excel(path: str) -> pd.DataFrame:
    print(f"[{datetime.now()}] Loading: {path}")
    df = pd.read_excel(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    print(f"[{datetime.now()}] Rows: {len(df)}")
    return df

def build_global_graph(df: pd.DataFrame) -> nx.DiGraph:
    print(f"[{datetime.now()}] Building ALL-drugs graph ...")
    G = nx.DiGraph()

    # caches to reduce duplicate nodes
    seen_drugs = set()
    seen_rx = set()
    seen_meta = set()        # (type, label)
    seen_diseases = set()
    seen_icd_full = set()
    seen_icd_root = set()
    invalid_total = set()

    for i, row in df.iterrows():
        if (i % 100) == 0:
            print(f"  ... row {i}/{len(df)}")

        # ---- Drug (L4)
        drug = _clean_str(row["Drug Name"])
        if not drug:
            continue
        if drug not in seen_drugs:
            _add_node(G, drug, type="Drug", label=drug)
            seen_drugs.add(drug)

        # attach common attrs (stored on Drug node)
        for k in ("Formulary","RxCUI","Brand Name","orig_approval_date"):
            v = _clean_str(row.get(k))
            if v is not None:
                G.nodes[drug][k.replace(" ","_").lower()] = v

        # ---- Approval Date node (L5)
        appr = _clean_str(row.get("orig_approval_date"))
        if appr:
            appr_node = f"Approval:{appr}"
            _add_node(G, appr_node, type="Approval_Date", label=appr)
            _add_edge(G, drug, appr_node, relation="APPROVED_ON")

        # ---- RxCUI node (L5)
        rxcui = _clean_str(row.get("RxCUI"))
        rxcui_node = None
        if rxcui:
            rxcui_node = f"RxCUI:{rxcui}"
            if rxcui_node not in seen_rx:
                _add_node(G, rxcui_node, type="RxCUI", label=rxcui)
                seen_rx.add(rxcui_node)
            _add_edge(G, drug, rxcui_node, relation="HAS_RXCUI")

        # ---- RxCUI children (L6) + also linked from Drug with requested relations
        def add_meta(col, node_type, drug_rel, rxcui_rel, prefix, listish=False):
            val = row.get(col)
            items = _parse_list_cell(val) if listish else ([_clean_str(val)] if _clean_str(val) else [])
            for it in items:
                lab = str(it)
                key = (node_type, lab)
                nid = f"{prefix}{lab}"
                if key not in seen_meta:
                    _add_node(G, nid, type=node_type, label=lab)
                    seen_meta.add(key)
                _add_edge(G, drug, nid, relation=drug_rel)
                if rxcui_node:
                    _add_edge(G, rxcui_node, nid, relation=rxcui_rel)

        add_meta("Brand Name", "Brand", "BRANDED_AS", "HAS_METADATA", "Brand:")
        add_meta("Formulary", "Formulary", "FORMULATED_AS", "HAS_METADATA", "Formulary:")
        for dose_col in [
            "Branded Dose Form Group", "Branded Drug Component", "Branded Drug or Pack",
            "Clinical Dose Form Group", "Clinical Drug Component", "Clinical Drug or Pack",
            "Dose Form Group"
        ]:
            add_meta(dose_col, "Dose_Form", "FORMULATED_AS", "HAS_METADATA", "DoseForm:")

        # Ingredients (listable)
        add_meta("Ingredient", "Ingredient", "CONTAINS", "HAS_METADATA", "Ingredient:", listish=True)
        add_meta("Precise Ingredient", "Precise_Ingredient", "CONTAINS", "HAS_METADATA", "PreciseIngredient:", listish=True)

        # ---- ICD hierarchy + diseases (L1/L2/L3)
        mapping = _parse_icd_mapping(row.get("may_treat_diseases_icd10"))
        if mapping:
            for disease, codes in mapping.items():
                if disease not in seen_diseases:
                    _add_node(G, disease, type="Disease", label=disease)
                    seen_diseases.add(disease)
                # drug ↔ disease
                _add_edge(G, drug, disease, relation="TREATS")
                _add_edge(G, disease, drug, relation="TREATED_BY")

                for full in set(codes or []):
                    if not ICD10_PATTERN.match(full):
                        invalid_total.add(full)
                        continue
                    if full not in seen_icd_full:
                        _add_node(G, full, type="ICD_Code", label=full)
                        seen_icd_full.add(full)
                    root = _icd_root(full)
                    if root and root not in seen_icd_root:
                        _add_node(G, root, type="ICD_Root", label=root)
                        seen_icd_root.add(root)
                    # edges (bidirectional where useful)
                    if root:
                        _add_edge(G, full, root, relation="BELONGS_TO")
                        _add_edge(G, root, full, relation="HAS_SUBCODE")
                    _add_edge(G, full, disease, relation="DIAGNOSED_AS")
                    _add_edge(G, disease, full, relation="HAS_CODE")

    if invalid_total:
        print(f"[WARN] Skipped invalid ICD-10 codes: {sorted(invalid_total)[:10]}{' ...' if len(invalid_total)>10 else ''}")

    print(f"[{datetime.now()}] Global graph: nodes={G.number_of_nodes()} edges={G.number_of_edges()}")
    return G

# ---------- exports ----------

def export_graphs(G: nx.DiGraph, outdir: str, tag: str):
    os.makedirs(outdir, exist_ok=True)
    graphml = os.path.join(outdir, f"{tag}.graphml")
    nx.write_graphml(G, graphml)
    print(f"Saved GraphML: {graphml}")

    pkl = os.path.join(outdir, f"{tag}.pkl")
    # nx.write_gpickle(G, pkl)
    # write_gpickle(G, pkl)
    with open(pkl, 'wb') as fh:
        pickle.dump(G, fh)
    print(f"Saved pickle:  {pkl}")

    js = os.path.join(outdir, f"{tag}.json")
    data = {
        "nodes": [{"id": n, **a} for n, a in G.nodes(data=True)],
        "edges": [{"source": u, "target": v, **d} for u, v, d in G.edges(data=True)],
    }
    with open(js, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Saved JSON:    {js}")

def export_neo4j_csv(G: nx.DiGraph, outdir: str, tag: str):
    nodes_csv = os.path.join(outdir, f"{tag}_nodes.csv")
    rels_csv = os.path.join(outdir, f"{tag}_relationships.csv")

    nodes = []
    for n, a in G.nodes(data=True):
        nodes.append({"id:ID": n, "label": a.get("label", n), "type:LABEL": a.get("type", "Node")})
    pd.DataFrame(nodes).to_csv(nodes_csv, index=False)

    rels = []
    for u, v, d in G.edges(data=True):
        rels.append({":START_ID": u, ":END_ID": v, ":TYPE": d.get("relation", "REL")})
    pd.DataFrame(rels).to_csv(rels_csv, index=False)

    print(f"Saved Neo4j CSVs: {nodes_csv}, {rels_csv}")

def export_adjacency(G: nx.DiGraph, outdir: str, tag: str):
    diseases = [n for n, a in G.nodes(data=True) if a.get("type") == "Disease"]
    drugs = [n for n, a in G.nodes(data=True) if a.get("type") == "Drug"]
    icd_full = [n for n, a in G.nodes(data=True) if a.get("type") == "ICD_Code"]

    # Disease x Drug
    mat = []
    for dz in diseases:
        row = {"Disease": dz}
        for dr in drugs:
            row[dr] = 1 if (G.has_edge(dr, dz) and G.edges[dr, dz].get("relation") == "TREATS") else 0
        mat.append(row)
    if mat:
        pd.DataFrame(mat).to_csv(os.path.join(outdir, f"{tag}_adj_disease_x_drug.csv"), index=False)

    # ICD x Disease
    mat2 = []
    for code in icd_full:
        row = {"ICD_Code": code}
        for dz in diseases:
            row[dz] = 1 if (G.has_edge(code, dz) and G.edges[code, dz].get("relation") == "DIAGNOSED_AS") else 0
        mat2.append(row)
    if mat2:
        pd.DataFrame(mat2).to_csv(os.path.join(outdir, f"{tag}_adj_icd_x_disease.csv"), index=False)

    print("Saved adjacency matrices.")

# ---------- visuals ----------

def snapshot_top_drugs(G: nx.DiGraph, outdir: str, tag: str, k: int = 3):
    """
    Quick static snapshots around the top-degree Drug nodes.
    We keep this lightweight to avoid plotting thousands of labels at once.
    """
    os.makedirs(outdir, exist_ok=True)
    drugs = [(n, G.degree(n)) for n, a in G.nodes(data=True) if a.get("type") == "Drug"]
    if not drugs:
        return
    drugs.sort(key=lambda x: x[1], reverse=True)
    top = [n for n, _ in drugs[:k]]

    for drug in top:
        # take 2-hop ego graph around this drug
        H = nx.ego_graph(G, drug, radius=2, undirected=False)
        pos = nx.spring_layout(H, k=0.4, seed=42)
        plt.figure(figsize=(10, 10))
        nx.draw_networkx_edges(H, pos, alpha=0.25, arrows=True, width=0.6)
        # color by type
        color_map = {
            "Drug": "#1f77b4", "Approval_Date": "#2ca02c", "RxCUI": "#2ca02c",
            "Disease": "#ff7f0e", "ICD_Code": "#d62728", "ICD_Root": "#9467bd",
            "Ingredient": "#8c564b", "Precise_Ingredient": "#8c564b",
            "Brand": "#8c564b", "Formulary": "#8c564b", "Dose_Form": "#8c564b",
        }
        for t, col in color_map.items():
            nlist = [n for n, a in H.nodes(data=True) if a.get("type") == t]
            if nlist:
                nx.draw_networkx_nodes(H, pos, nodelist=nlist, node_color=col, node_size=130, alpha=0.9)
        # labels (trim if huge)
        if H.number_of_nodes() <= 250:
            labels = {n: H.nodes[n].get("label", n) for n in H.nodes()}
            nx.draw_networkx_labels(H, pos, labels=labels, font_size=8)

        plt.axis("off")
        plt.title(f"{drug} ego snapshot (2-hop)")
        png = os.path.join(outdir, f"{tag}_snapshot_{drug.replace(' ','_')}.png")
        plt.tight_layout()
        plt.savefig(png, dpi=200)
        plt.close()
        print(f"Saved snapshot: {png}")

def export_pyvis(G: nx.DiGraph, out_html: str):
    if not _PYVIS:
        print("[WARN] pyvis not installed; skipping interactive HTML.")
        return
    net = Network(height="850px", width="100%", directed=True, notebook=False, bgcolor="#ffffff", font_color="#222")
    net.barnes_hut()

    color_map = {
        "Drug": "#1f77b4", "Approval_Date": "#2ca02c", "RxCUI": "#2ca02c",
        "Disease": "#ff7f0e", "ICD_Code": "#d62728", "ICD_Root": "#9467bd",
        "Ingredient": "#8c564b", "Precise_Ingredient": "#8c564b",
        "Brand": "#8c564b", "Formulary": "#8c564b", "Dose_Form": "#8c564b",
    }
    for n, a in G.nodes(data=True):
        t = a.get("type", "Node")
        net.add_node(n, label=a.get("label", n), color=color_map.get(t, "#7f7f7f"), title=t)
    for u, v, d in G.edges(data=True):
        net.add_edge(u, v, title=d.get("relation", ""))

    # define the options as a Python dict
    options = {
        "edges": {
            "arrows": {"to": {"enabled": True, "scaleFactor": 0.7}},
            "smooth": {"type": "dynamic"}
        },
        "physics": {"stabilization": True}
    }

    # convert to JSON and set the options
    net.set_options(json.dumps(options))
    # net.set_options("""
    #   const options = {
    #     edges: { arrows: { to: {enabled: true, scaleFactor: 0.7} }, smooth: { type: 'dynamic' } },
    #     physics: { stabilization: true }
    #   }""")
    net.show(out_html)
    print(f"Saved interactive HTML: {out_html}")

# ---------- queries & summary ----------

def diseases_for_drug(G: nx.DiGraph, drug: str) -> List[str]:
    return sorted([v for u, v, data in G.out_edges(drug, data=True) if data.get("relation") == "TREATS"])

def drugs_for_icd_root(G: nx.DiGraph, root_code: str) -> List[str]:
    root = root_code.strip().upper()
    if root not in G:
        return []
    drugs = set()
    fulls = [v for u, v, d in G.out_edges(root, data=True) if d.get("relation") == "HAS_SUBCODE"]
    for code in fulls:
        dzs = [v for u, v, d in G.out_edges(code, data=True) if d.get("relation") == "DIAGNOSED_AS"]
        for dz in dzs:
            for u, v, d in G.in_edges(dz, data=True):
                if d.get("relation") == "TREATS" and G.nodes[u].get("type") == "Drug":
                    drugs.add(u)
    return sorted(drugs)

def ingredient_frequency(G: nx.DiGraph) -> Counter:
    c = Counter()
    for u, v, d in G.edges(data=True):
        if d.get("relation") == "CONTAINS" and G.nodes[v].get("type") in ("Ingredient","Precise_Ingredient"):
            c[G.nodes[v].get("label", v)] += 1
    return c

def approval_timeline(G: nx.DiGraph, drug: str) -> List[str]:
    return sorted([G.nodes[v].get("label", v).replace("Approval:","")
                   for u, v, d in G.out_edges(drug, data=True)
                   if d.get("relation") == "APPROVED_ON"])

def summarize(G: nx.DiGraph):
    types = Counter([a.get("type","Node") for _, a in G.nodes(data=True)])
    rels = Counter([d.get("relation","REL") for _, _, d in G.edges(data=True)])
    print("\n=== SUMMARY ===")
    print("Nodes by type:")
    for t, n in sorted(types.items()):
        print(f"  {t:20s} {n}")
    print("Edges by relation:")
    for r, n in sorted(rels.items()):
        print(f"  {r:20s} {n}")
    print("================\n")

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description="Build a SINGLE global pharma knowledge graph from Excel.")
    ap.add_argument("--excel", required=True, help="Path to rxnav_with_fda_with_orig.xlsx")
    ap.add_argument("--outdir", default="kg_all", help="Output directory")
    ap.add_argument("--nohtml", action="store_true", help="Skip interactive HTML")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = load_excel(args.excel)

    # Global, no filtering
    G = build_global_graph(df)

    tag = "ALL_DRUGS_KG"
    export_graphs(G, args.outdir, tag)
    export_neo4j_csv(G, args.outdir, tag)
    export_adjacency(G, args.outdir, tag)

    # Optional visuals
    snapshot_top_drugs(G, args.outdir, tag, k=3)
    if not args.nohtml:
        export_pyvis(G, os.path.join(args.outdir, f"{tag}.html"))

    summarize(G)

    # Example console queries (printed hints)
    print("Example query hints:")
    print("  - diseases_for_drug(G, '<Drug Name>')")
    print("  - drugs_for_icd_root(G, 'F11')")
    print("  - ingredient_frequency(G').most_common(10)")
    print("  - approval_timeline(G, '<Drug Name>')")


if __name__ == "__main__":
    main()
