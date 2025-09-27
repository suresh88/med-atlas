"""
Dash/Cytoscape Knowledge Graph Viewer
------------------------------------

This script launches an interactive web application using Dash and
dash_cytoscape to visualise a pharmaceutical knowledge graph.

Features:
* Loads a graph from a GraphML or JSON file.
* Displays the graph in a Cytoscape component with a force-directed layout.
* Colours nodes by type (Drug, Disease, ICD codes, etc.).
* Clicking a node highlights its immediate neighbours.

Usage:

    python knowledge_graph_dash_app.py --graph path/to/ALL_DRUGS_KG.graphml

Dependencies:

    pip install dash dash-cytoscape networkx

Once running, open your browser at http://127.0.0.1:8050/ to explore the graph.
"""

import argparse
import json
import networkx as nx

import dash
from dash import html
from dash.dependencies import Input, Output
import dash_cytoscape as cyto

# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------
def load_graph(path: str) -> nx.DiGraph:
    """Load a graph from a GraphML or JSON file."""
    if path.lower().endswith(".graphml"):
        return nx.read_graphml(path)
    elif path.lower().endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        G = nx.DiGraph()
        for node in data.get("nodes", []):
            node_id = node.get("id")
            attrs = {k: v for k, v in node.items() if k != "id"}
            G.add_node(node_id, **attrs)
        for edge in data.get("edges", []):
            src = edge.get("source")
            tgt = edge.get("target")
            attrs = {k: v for k, v in edge.items() if k not in ("source", "target")}
            G.add_edge(src, tgt, **attrs)
        return G
    else:
        raise ValueError(f"Unsupported file format: {path}")

# ----------------------------------------------------------------------
# Conversion to Cytoscape format
# ----------------------------------------------------------------------
def to_cytoscape_elements(G: nx.DiGraph) -> list:
    """Convert a NetworkX graph into cytoscape.js element dictionaries."""
    elements = []
    for node_id, attrs in G.nodes(data=True):
        elements.append({
            "data": {
                "id": node_id,
                "label": attrs.get("label", node_id),
                "type": attrs.get("type", "")
            }
        })
    for u, v, attrs in G.edges(data=True):
        elements.append({
            "data": {
                "source": u,
                "target": v,
                "label": attrs.get("relation", attrs.get("label", ""))
            }
        })
    return elements

# ----------------------------------------------------------------------
# Stylesheet
# ----------------------------------------------------------------------
def build_base_stylesheet() -> list:
    """Return a base stylesheet mapping node types to colours."""
    type_colour = {
        "Drug": "#1f77b4",
        "Approval_Date": "#2ca02c",
        "RxCUI": "#2ca02c",
        "Disease": "#ff7f0e",
        "ICD_Code": "#d62728",
        "ICD_Root": "#9467bd",
        "Ingredient": "#8c564b",
        "Precise_Ingredient": "#8c564b",
        "Brand": "#8c564b",
        "Formulary": "#8c564b",
        "Dose_Form": "#8c564b",
    }
    stylesheet = [
        {
            'selector': 'node',
            'style': {
                'label': 'data(label)',
                'font-size': '12px',
                'text-valign': 'center',
                'text-halign': 'center',
                'width': '30px',
                'height': '30px',
            }
        },
        {
            'selector': 'edge',
            'style': {
                'curve-style': 'bezier',
                'target-arrow-shape': 'triangle',
                'target-arrow-color': '#bbb',
                'line-color': '#bbb',
                'opacity': 0.8,
            }
        }
    ]
    for typ, colour in type_colour.items():
        stylesheet.append({
            'selector': f'[type = "{typ}"]',
            'style': {'background-color': colour}
        })
    return stylesheet

def highlight_stylesheet(base: list, G: nx.DiGraph, selected: str) -> list:
    """
    Return a modified stylesheet highlighting the selected node and its neighbours
    while fading all others.
    """
    highlight_nodes = {selected}
    if selected in G:
        highlight_nodes.update(G.successors(selected))
        highlight_nodes.update(G.predecessors(selected))

    styled = list(base)
    styled.append({'selector': 'node', 'style': {'opacity': 0.2}})
    styled.append({'selector': 'edge', 'style': {'opacity': 0.1}})
    for node_id in highlight_nodes:
        styled.append({
            'selector': f'[id = "{node_id}"]',
            'style': {
                'opacity': 1.0,
                'background-color': '#FFD700',
                'border-width': '2px',
                'border-color': '#333'
            }
        })
    for u, v in G.edges():
        if u in highlight_nodes and v in highlight_nodes:
            styled.append({
                'selector': f'edge[source = "{u}"][target = "{v}"]',
                'style': {
                    'opacity': 1.0,
                    'line-color': '#FFD700',
                    'target-arrow-color': '#FFD700',
                    'width': 3
                }
            })
    return styled

# ----------------------------------------------------------------------
# Dash app
# ----------------------------------------------------------------------
def create_app(G: nx.DiGraph) -> dash.Dash:
    """Create a Dash app for exploring the given graph."""
    elements = to_cytoscape_elements(G)
    base_stylesheet = build_base_stylesheet()

    app = dash.Dash(__name__)
    app.layout = html.Div([
        html.H2("Knowledge Graph Explorer"),
        cyto.Cytoscape(
            id='cytoscape',
            elements=elements,
            layout={'name': 'cose'},
            stylesheet=base_stylesheet,
            style={'width': '100%', 'height': '800px'},
        ),
    ])

    @app.callback(
        Output('cytoscape', 'stylesheet'),
        [Input('cytoscape', 'tapNodeData')]
    )
    def update_stylesheet(node_data):
        if not node_data:
            return base_stylesheet
        node_id = node_data['id']
        return highlight_stylesheet(base_stylesheet, G, node_id)

    return app

# ----------------------------------------------------------------------
# Command-line entry
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Run an interactive Dash app to explore a knowledge graph.")
    parser.add_argument('--graph', required=True, help='Path to a GraphML or JSON file containing the graph.')
    parser.add_argument('--port', type=int, default=8050, help='Port to run the web server on (default 8050).')
    args = parser.parse_args()

    G = load_graph(args.graph)
    app = create_app(G)
    app.run(debug=True, port=args.port)

if __name__ == '__main__':
    main()
