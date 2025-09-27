import pandas as pd
import ast
import re
from datetime import datetime
from collections import defaultdict, Counter
import json
from xml.etree.ElementTree import Element, SubElement, tostring
import xml.dom.minidom

import argparse
import os

"""
This script reads pharmaceutical data from an Excel file and constructs a
comprehensive knowledge graph capturing relationships between ICD‑10
codes, diseases, drugs, RxCUI identifiers, brand names, ingredients,
precise ingredients and approval dates.  The script then exports this
graph in GraphML format, produces an interactive HTML visualization using
D3.js, and generates a Markdown report summarising key statistics and
providing example queries.

Steps performed:

1. **Data ingestion**: Load selected columns from the Excel file.
2. **Graph construction**: Iterate through the rows and build
   hierarchical nodes and edges representing ICD‑10 structure,
   drug–disease treatment relationships, and medication metadata.
3. **GraphML export**: Write the graph to a GraphML file with
   attributes for node types and relationships for edges.
4. **Interactive visualization**: Generate an HTML file that uses D3.js
   to display a force‑directed network.  Node colours are derived
   from their types.
5. **Analysis & report**: Compute summary statistics (counts of node
   and edge types, most connected diseases, approval timeline,
   ingredient frequencies) and write a Markdown report.

The generated files include:
  * pharma_knowledge_graph.graphml – the graph in GraphML format.
  * pharma_knowledge_graph.html – an interactive HTML visualization.
  * pharma_report.md – a human‑readable report summarising the graph.

To run this script, ensure that pandas is installed and that the
provided Excel file exists at the specified path.
"""


def parse_approval_date(value):
    """Convert various date formats to ISO strings or return None."""
    if pd.isna(value):
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    try:
        return pd.to_datetime(value).date().isoformat()
    except Exception:
        return None


def build_graph(df):
    """Construct nodes and edges from the input DataFrame.

    Returns:
        nodes (dict): mapping node id to attribute dict
        edges (list): list of edge dicts with keys 'source', 'target', 'type'
        node_type_counts (defaultdict): counts of nodes by type
        edge_type_counts (defaultdict): counts of edges by relationship
    """
    nodes = {}
    edges = []
    node_type_counts = defaultdict(int)
    edge_type_counts = defaultdict(int)

    def add_node(node_id, node_type, **attrs):
        if not node_id:
            return
        if node_id not in nodes:
            nodes[node_id] = {'id': node_id, 'type': node_type, **attrs}
            node_type_counts[node_type] += 1
        else:
            # fill in missing attributes if any
            for key, val in attrs.items():
                if key not in nodes[node_id] or nodes[node_id][key] is None:
                    nodes[node_id][key] = val

    def add_edge(src, tgt, rel):
        if not src or not tgt:
            return
        edges.append({'source': src, 'target': tgt, 'type': rel})
        edge_type_counts[rel] += 1

    for _, row in df.iterrows():
        drug_name = str(row['Drug Name']).strip() if pd.notna(row['Drug Name']) else None
        rx_cui = str(row['RxCUI']).strip() if pd.notna(row['RxCUI']) else None
        # split multi‑valued fields on common delimiters
        brand_names = []
        if pd.notna(row['Brand Name']):
            brand_names = re.split(r'[\n;,]+', str(row['Brand Name']))
            brand_names = [bn.strip() for bn in brand_names if bn.strip()]
        ingredients = []
        if pd.notna(row['Ingredient']):
            ingredients = re.split(r'[\n;,]+', str(row['Ingredient']))
            ingredients = [ing.strip() for ing in ingredients if ing.strip()]
        precise_ings = []
        if pd.notna(row['Precise Ingredient']):
            precise_ings = re.split(r'[\n;,]+', str(row['Precise Ingredient']))
            precise_ings = [pi.strip() for pi in precise_ings if pi.strip()]
        # parse diseases and codes
        diseases = []
        if pd.notna(row['may_treat_diseases']):
            try:
                diseases = ast.literal_eval(row['may_treat_diseases'])
            except Exception:
                diseases = row['may_treat_diseases'] if isinstance(row['may_treat_diseases'], list) else []
        codes_map = []
        if pd.notna(row['may_treat_diseases_icd10']):
            try:
                codes_map = ast.literal_eval(row['may_treat_diseases_icd10'])
            except Exception:
                codes_map = []
        disease_to_codes = {}
        for item in codes_map:
            if isinstance(item, dict):
                for d, codes in item.items():
                    disease_to_codes[d] = codes
        if not diseases and disease_to_codes:
            diseases = list(disease_to_codes.keys())
        # add drug node
        add_node(
            drug_name,
            'Drug',
            rx_cui=rx_cui,
            brand_names=brand_names,
            ingredients=ingredients,
            precise_ingredients=precise_ings,
            approval_date=parse_approval_date(row['orig_approval_date']),
        )
        # connect drug to diseases and codes
        for disease in diseases:
            d_name = disease.strip()
            add_node(d_name, 'Disease')
            add_edge(drug_name, d_name, 'TREATS')
            for full_code in disease_to_codes.get(disease, []):
                full_code = str(full_code).strip()
                if not full_code:
                    continue
                root_code = full_code.split('.')[0] if '.' in full_code else full_code[:3]
                add_node(root_code, 'ICD10_Root')
                add_node(full_code, 'ICD10_Full', root_code=root_code)
                add_edge(full_code, root_code, 'BELONGS_TO')
                add_edge(d_name, full_code, 'DIAGNOSED_AS')
        # RxCUI relationships
        if rx_cui and rx_cui != 'nan':
            add_node(rx_cui, 'RxCUI')
            add_edge(drug_name, rx_cui, 'HAS_RXCUI')
            for bn in brand_names:
                add_node(bn, 'BrandName')
                add_edge(rx_cui, bn, 'BRANDED_AS')
            for ing in ingredients:
                add_node(ing, 'Ingredient')
                add_edge(rx_cui, ing, 'CONTAINS')
            for pi in precise_ings:
                add_node(pi, 'PreciseIngredient')
                add_edge(rx_cui, pi, 'CONTAINS_PRECISE')
        # approval date node
        ap_date = parse_approval_date(row['orig_approval_date'])
        if ap_date:
            add_node(ap_date, 'ApprovalDate')
            add_edge(drug_name, ap_date, 'APPROVED_ON')
    return nodes, edges, node_type_counts, edge_type_counts


def export_graphml(nodes, edges, node_type_counts, edge_type_counts, out_path):
    """Write the graph to a GraphML file with node and edge attributes."""
    # collect all node attributes except id and type
    attr_names = set()
    for node_data in nodes.values():
        for attr in node_data:
            if attr not in ('id', 'type'):
                attr_names.add(attr)
    graphml = Element('graphml', {
        'xmlns': 'http://graphml.graphdrawing.org/xmlns',
        'xmlns:xsi': 'http://www.w3.org/2001/XMLSchema-instance',
        'xsi:schemaLocation': 'http://graphml.graphdrawing.org/xmlns http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd',
    })
    key_map = {}
    key_index = 0
    # key for node type
    k_type = SubElement(graphml, 'key', {
        'id': f'k{key_index}', 'for': 'node', 'attr.name': 'type', 'attr.type': 'string'
    })
    key_map['type'] = f'k{key_index}'
    key_index += 1
    # keys for other node attributes
    for attr in sorted(attr_names):
        k_attr = SubElement(graphml, 'key', {
            'id': f'k{key_index}', 'for': 'node', 'attr.name': attr, 'attr.type': 'string'
        })
        key_map[attr] = f'k{key_index}'
        key_index += 1
    # key for edge relationship
    k_edge = SubElement(graphml, 'key', {
        'id': f'k{key_index}', 'for': 'edge', 'attr.name': 'relationship', 'attr.type': 'string'
    })
    edge_key = f'k{key_index}'
    key_index += 1
    graph_el = SubElement(graphml, 'graph', {'id': 'G', 'edgedefault': 'directed'})
    # write nodes
    for node_id, node_data in nodes.items():
        n_el = SubElement(graph_el, 'node', {'id': node_id})
        # type
        t_data = SubElement(n_el, 'data', {'key': key_map['type']})
        t_data.text = node_data['type']
        # other attributes
        for attr in attr_names:
            val = node_data.get(attr)
            if val is None:
                continue
            if isinstance(val, list):
                txt = ';'.join(str(v) for v in val)
            else:
                txt = str(val)
            d_el = SubElement(n_el, 'data', {'key': key_map[attr]})
            d_el.text = txt
    # write edges
    for idx, edge in enumerate(edges):
        e_el = SubElement(graph_el, 'edge', {'id': f'e{idx}', 'source': edge['source'], 'target': edge['target']})
        rel_data = SubElement(e_el, 'data', {'key': edge_key})
        rel_data.text = edge['type']
    # pretty print XML
    xmlstr = xml.dom.minidom.parseString(tostring(graphml)).toprettyxml(indent="  ")
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(xmlstr)


def export_html(nodes, edges, out_path):
    """Generate an interactive HTML visualization using D3.js."""
    # assign indices for nodes
    nodes_list = []
    index_map = {}
    for i, (node_id, node_data) in enumerate(nodes.items()):
        index_map[node_id] = i
        nodes_list.append({'id': node_id, 'label': node_id, 'type': node_data['type']})
    links_list = []
    for e in edges:
        if e['source'] in index_map and e['target'] in index_map:
            links_list.append({'source': index_map[e['source']], 'target': index_map[e['target']], 'type': e['type']})
    # define colours for node types
    colour_map = {
        'ICD10_Root': '#66C2A5',
        'ICD10_Full': '#E78AC3',
        'Disease': '#8DA0CB',
        'Drug': '#FC8D62',
        'RxCUI': '#A6D854',
        'BrandName': '#FFD92F',
        'Ingredient': '#E5C494',
        'PreciseIngredient': '#B3B3B3',
        'ApprovalDate': '#FB8072',
    }
    html_content = f"""
<!DOCTYPE html>
<html lang='en'>
<head>
    <meta charset='UTF-8'>
    <title>Pharma Knowledge Graph</title>
    <style>
        .node circle {{ stroke: #fff; stroke-width: 1.5px; }}
        .link {{ stroke: #999; stroke-opacity: 0.6; }}
        .tooltip {{
            position: absolute;
            text-align: center;
            padding: 4px;
            font: 12px sans-serif;
            background: lightgrey;
            border: 1px solid #ddd;
            border-radius: 4px;
            pointer-events: none;
        }}
    </style>
</head>
<body>
    <h2>Pharma Knowledge Graph</h2>
    <div id='graph'></div>
    <script src='https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.2/d3.min.js'></script>
    <script>
        var nodes = {json.dumps(nodes_list)};
        var links = {json.dumps(links_list)};
        var colourMap = {json.dumps(colour_map)};
        var width = 1200;
        var height = 800;
        var svg = d3.select('#graph').append('svg')
            .attr('width', width)
            .attr('height', height);
        var tooltip = d3.select('body').append('div')
            .attr('class', 'tooltip')
            .style('opacity', 0);
        var simulation = d3.forceSimulation(nodes)
            .force('link', d3.forceLink(links).id(function(d) {{ return d.id; }}).distance(50))
            .force('charge', d3.forceManyBody().strength(-50))
            .force('center', d3.forceCenter(width / 2, height / 2));
        var link = svg.append('g')
            .attr('class', 'links')
            .selectAll('line')
            .data(links)
            .enter().append('line')
            .attr('stroke-width', 1);
        var node = svg.append('g')
            .attr('class', 'nodes')
            .selectAll('g')
            .data(nodes)
            .enter().append('g')
            .call(d3.drag()
                .on('start', dragstarted)
                .on('drag', dragged)
                .on('end', dragended));
        node.append('circle')
            .attr('r', 4)
            .attr('fill', function(d) {{ return colourMap[d.type] || '#ccc'; }});
        node.append('title')
            .text(function(d) {{ return d.id + ' (' + d.type + ')'; }});
        node.on('mouseover', function(event, d) {{
            tooltip.transition().duration(200).style('opacity', .9);
            tooltip.html(d.id + ' (' + d.type + ')')
                .style('left', (event.pageX + 5) + 'px')
                .style('top', (event.pageY - 28) + 'px');
        }})
        .on('mouseout', function(event, d) {{
            tooltip.transition().duration(500).style('opacity', 0);
        }});
        simulation.nodes(nodes).on('tick', ticked);
        simulation.force('link').links(links);
        function ticked() {{
            link.attr('x1', function(d) {{ return d.source.x; }})
                .attr('y1', function(d) {{ return d.source.y; }})
                .attr('x2', function(d) {{ return d.target.x; }})
                .attr('y2', function(d) {{ return d.target.y; }});
            node.attr('transform', function(d) {{ return 'translate(' + d.x + ',' + d.y + ')'; }});
        }}
        function dragstarted(event, d) {{
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
        }}
        function dragged(event, d) {{
            d.fx = event.x;
            d.fy = event.y;
        }}
        function dragended(event, d) {{
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
        }}
    </script>
</body>
</html>
"""
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html_content)


def generate_report(nodes, edges, node_type_counts, edge_type_counts, out_path):
    """Compute simple analytics and write a markdown report."""
    # Build useful mappings for analysis
    disease_to_drugs = defaultdict(set)
    drug_to_diseases = defaultdict(set)
    code_to_diseases = defaultdict(set)
    full_to_root = {}
    for e in edges:
        rel = e['type']
        if rel == 'TREATS':
            disease_to_drugs[e['target']].add(e['source'])
            drug_to_diseases[e['source']].add(e['target'])
        elif rel == 'DIAGNOSED_AS':
            code_to_diseases[e['target']].add(e['source'])
        elif rel == 'BELONGS_TO':
            full_to_root[e['source']] = e['target']
    root_to_diseases = defaultdict(set)
    for full_code, root in full_to_root.items():
        for disease in code_to_diseases.get(full_code, []):
            root_to_diseases[root].add(disease)
    # most connected diseases
    max_conn = 0
    most_connected = []
    for disease, drugs in disease_to_drugs.items():
        count = len(drugs)
        if count > max_conn:
            max_conn = count
            most_connected = [disease]
        elif count == max_conn:
            most_connected.append(disease)
    # approval timeline by year
    year_counts = Counter()
    for node_id, data in nodes.items():
        if data['type'] == 'Drug' and data.get('approval_date'):
            year = data['approval_date'].split('-')[0]
            year_counts[year] += 1
    # ingredient frequencies
    ingredient_counts = Counter()
    for node_id, data in nodes.items():
        if data['type'] == 'Drug':
            for ing in data.get('ingredients', []):
                ingredient_counts[ing] += 1
    # build report content
    lines = []
    lines.append('# Pharma Knowledge Graph Summary')
    lines.append('\n## Overview')
    lines.append(f'The knowledge graph contains **{len(nodes)}** nodes and **{len(edges)}** edges.')
    lines.append('\n### Node counts by type')
    for ntype, count in sorted(node_type_counts.items()):
        lines.append(f'- **{ntype}**: {count}')
    lines.append('\n### Edge counts by relationship')
    for rel, count in sorted(edge_type_counts.items()):
        lines.append(f'- **{rel}**: {count}')
    lines.append('\n## Key Insights')
    lines.append(f'- **Most connected diseases (treated by {max_conn} drugs)**: ' + ', '.join(most_connected))
    lines.append('- **Drug approval timeline**: Top approval years include:')
    for year, count in year_counts.most_common(10):
        lines.append(f'  - {year}: {count} approvals')
    lines.append('- **Ingredient frequency**: The most common ingredients:')
    for ing, count in ingredient_counts.most_common(10):
        lines.append(f'  - {ing}: {count} drugs')
    # simple query demonstration
    lines.append('\n## Sample Queries')
    sample_drugs = list(drug_to_diseases.keys())[:3]
    for drug in sample_drugs:
        diseases = ', '.join(drug_to_diseases[drug])
        lines.append(f'- **{drug}** treats: {diseases}')
    if root_to_diseases:
        root_code = next(iter(root_to_diseases))
        drugs_list = set()
        for disease in root_to_diseases[root_code]:
            drugs_list.update(disease_to_drugs[disease])
        sample = ', '.join(list(drugs_list)[:10])
        lines.append(f'- Drugs treating ICD‑10 category **{root_code}**: {sample} ...')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def main():
    """
    Command‑line entry point for generating the pharmaceutical knowledge graph.

    Expects two positional arguments: the path to an input Excel file and
    a directory where output files will be written. The script will
    create the output directory if it does not already exist and will
    write three files there:

      * pharma_knowledge_graph.graphml – the graph in GraphML format
      * pharma_knowledge_graph.html – an interactive D3.js visualization
      * pharma_report.md – a summary report in Markdown format
    """
    parser = argparse.ArgumentParser(
        description='Generate a pharmaceutical knowledge graph and associated files.'
    )
    # Support both optional flags and positional arguments for flexibility.  The
    # user may specify the input Excel file and output directory as either
    # positional arguments or via --input and --output flags.  If both forms
    # are provided, the flags take precedence.
    parser.add_argument('input', nargs='?', help='Path to the input Excel file (positional)')
    parser.add_argument('output', nargs='?', help='Output directory for generated files (positional)')
    parser.add_argument('--input', dest='input_flag', help='Path to the input Excel file (flag)')
    parser.add_argument('--output', dest='output_flag', help='Output directory for generated files (flag)')
    args = parser.parse_args()
    # Determine the actual input and output values, preferring flags over
    # positional arguments if both are provided.
    input_path = args.input_flag or args.input
    output_dir = args.output_flag or args.output
    if not input_path or not output_dir:
        parser.error('Both input and output must be provided either as positional arguments or via --input/--output flags.')
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    # Load only the necessary columns from the input Excel file
    df = pd.read_excel(input_path, usecols=[
        'Drug Name',
        'RxCUI',
        'Brand Name',
        'Ingredient',
        'Precise Ingredient',
        'may_treat_diseases',
        'may_treat_diseases_icd10',
        'orig_approval_date',
    ])
    # Build the graph
    nodes, edges, node_counts, edge_counts = build_graph(df)
    # Compose output file paths
    graphml_path = os.path.join(output_dir, 'pharma_knowledge_graph.graphml')
    html_path = os.path.join(output_dir, 'pharma_knowledge_graph.html')
    report_path = os.path.join(output_dir, 'pharma_report.md')
    # Export the graph and ancillary files
    export_graphml(nodes, edges, node_counts, edge_counts, graphml_path)
    export_html(nodes, edges, html_path)
    generate_report(nodes, edges, node_counts, edge_counts, report_path)
    print('Generation complete. Files saved to', output_dir)


if __name__ == '__main__':
    main()