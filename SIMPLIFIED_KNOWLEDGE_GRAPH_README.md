# Simplified Knowledge Graph

## Overview
The updated `knowledge_graph_builder6.py` now creates a simplified knowledge graph with only the following node types:

## Node Types
1. **Drug** - The main drug names from the pharmaceutical data
2. **RxCUI** - RxCUI identifiers for drugs
3. **Brand** - Brand names of drugs
4. **Ingredient** - Active ingredients in drugs
5. **Precise_Ingredient** - More specific ingredient information
6. **Disease** - Diseases that drugs may treat (from both `may_treat_diseases` and `may_treat_diseases_icd10`)
7. **Approval_Date** - Original FDA approval dates

## Relationships
- **Drug → RxCUI**: `HAS_RXCUI`
- **Drug → Brand**: `BRANDED_AS`
- **Drug → Ingredient**: `CONTAINS`
- **Drug → Precise_Ingredient**: `CONTAINS`
- **Drug → Disease**: `TREATS`
- **Drug → Approval_Date**: `APPROVED_ON`

## Removed Components
The following node types and their relationships have been removed:
- ICD_Code and ICD_Root (ICD-10 hierarchy)
- Formulary information
- All dose form categories (Branded/Clinical/Generic Dose Forms)
- Complex metadata relationships

## Running the Code

### Generate the Knowledge Graph
```bash
conda activate healthcare
python src/knowledge_graph/knowledge_graph_builder6.py --excel data/output/rxnav_with_fda_sample.xlsx --outdir data/output/kg_simplified
```

### Visualize with Dash
```bash
conda activate healthcare
python src/knowledge_graph/dash_visualizer.py --graph data/output/kg_simplified/ALL_DRUGS_KG.json --port 8051
```

Then open http://127.0.0.1:8051/ in your browser.

## Output Files
The script generates:
- GraphML format (`.graphml`)
- JSON format (`.json`) 
- Pickle format (`.pkl`)
- Neo4j CSV files (`_nodes.csv`, `_relationships.csv`)
- Adjacency matrix (Disease x Drug)
- Interactive HTML visualization
- Static PNG snapshots of top drugs

## Example Graph Statistics
For the sample dataset:
- 93 total nodes
- 101 total edges
- 14 drugs, 14 diseases, 18 ingredients, 7 precise ingredients
- 14 brand names, 14 RxCUI entries, 12 approval dates