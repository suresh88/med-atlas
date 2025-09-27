import pandas as pd
import graphistry
from typing import Dict, List, Optional

def create_healthcare_graphistry_viz(nodes: List[Dict], edges: List[Dict], 
                                   graphistry_credentials: Dict[str, str]):
    """
    Simple, working Graphistry visualization using basic API methods only
    
    Args:
        nodes: List of node dictionaries with healthcare data
        edges: List of edge dictionaries with relationships
        graphistry_credentials: Graphistry login credentials
    """
    
    # Setup Graphistry
    graphistry.register(
        api=3,
        username=graphistry_credentials['username'],
        password=graphistry_credentials['password']
    )
    
    # Convert to DataFrames
    nodes_df = pd.DataFrame(nodes)
    edges_df = pd.DataFrame(edges)
    
    print(f"Processing {len(nodes_df)} nodes and {len(edges_df)} edges")
    
    # Ensure required columns exist
    if 'id' not in nodes_df.columns:
        nodes_df['id'] = nodes_df.index.astype(str)
    
    # Standardize edge columns
    if 'source' not in edges_df.columns:
        edges_df['source'] = edges_df.iloc[:, 0]
    if 'target' not in edges_df.columns:
        if 'destination' in edges_df.columns:
            edges_df['target'] = edges_df['destination']
        else:
            edges_df['target'] = edges_df.iloc[:, 1]
    
    # Add display labels for nodes
    if 'name' in nodes_df.columns:
        nodes_df['label'] = nodes_df['name']
    elif 'label' not in nodes_df.columns:
        nodes_df['label'] = nodes_df['id']
    
    # Create base Graphistry object - using only basic methods
    try:
        g = graphistry.bind(source='source', destination='target')
        g = g.edges(edges_df)
        
        # Add nodes if available
        if not nodes_df.empty:
            g = g.bind(node='id')
            g = g.nodes(nodes_df)
        
        print("Base graph created successfully")
        
        # Apply basic styling using bind() method
        
        # Color by node type if available
        if 'node_type' in nodes_df.columns:
            g = g.bind(point_color='node_type')
            print("Applied node type coloring")
        
        # Color by formulary tier if available
        elif 'formulary_tier' in nodes_df.columns:
            g = g.bind(point_color='formulary_tier')
            print("Applied formulary tier coloring")
        
        # Size by importance metrics if available
        if 'prescription_volume' in nodes_df.columns:
            g = g.bind(point_size='prescription_volume')
            print("Applied prescription volume sizing")
        elif 'cost' in nodes_df.columns:
            g = g.bind(point_size='cost')
            print("Applied cost-based sizing")
        
        # Edge coloring by relationship type
        if 'relationship' in edges_df.columns:
            g = g.bind(edge_color='relationship')
            print("Applied relationship type edge coloring")
        
        # Add labels
        g = g.bind(point_title='label')
        
        return g
        
    except Exception as e:
        print(f"Error in graph creation: {e}")
        # Fallback: Create minimal graph
        g = graphistry.bind(source='source', destination='target').edges(edges_df)
        return g

def create_enhanced_healthcare_viz(nodes: List[Dict], edges: List[Dict], 
                                 graphistry_credentials: Dict[str, str]):
    """
    Enhanced version with data preprocessing for better healthcare visualization
    """
    
    # Setup Graphistry
    graphistry.register(
        api=3,
        username=graphistry_credentials['username'],
        password=graphistry_credentials['password']
    )
    
    # Convert to DataFrames
    nodes_df = pd.DataFrame(nodes)
    edges_df = pd.DataFrame(edges)
    
    # Data preprocessing for healthcare context
    
    # 1. Create meaningful labels
    def create_node_label(row):
        if row.get('name'):
            label = row['name']
        else:
            label = str(row.get('id', 'Unknown'))
        
        # Add context for healthcare nodes
        if row.get('node_type') == 'drug':
            if row.get('formulary_tier'):
                label += f" (Tier {row['formulary_tier']})"
        elif row.get('node_type') == 'condition':
            if row.get('icd_code'):
                label += f" [{row['icd_code']}]"
        
        return label
    
    nodes_df['display_label'] = nodes_df.apply(create_node_label, axis=1)
    
    # 2. Create categorical colors for better visualization
    if 'node_type' in nodes_df.columns:
        # Map node types to numeric values for coloring
        type_mapping = {
            'drug': 1,
            'condition': 2,
            'provider': 3,
            'facility': 4,
            'therapeutic_class': 5
        }
        nodes_df['type_numeric'] = nodes_df['node_type'].map(type_mapping).fillna(0)
    
    # 3. Create hover information
    def create_hover_text(row):
        hover_parts = [f"ID: {row.get('id', 'N/A')}"]
        
        if row.get('node_type'):
            hover_parts.append(f"Type: {row['node_type']}")
        if row.get('formulary_tier'):
            hover_parts.append(f"Formulary Tier: {row['formulary_tier']}")
        if row.get('therapeutic_class'):
            hover_parts.append(f"Class: {row['therapeutic_class']}")
        if row.get('prescription_volume'):
            hover_parts.append(f"Rx Volume: {row['prescription_volume']}")
        
        return " | ".join(hover_parts)
    
    nodes_df['hover_info'] = nodes_df.apply(create_hover_text, axis=1)
    
    # 4. Process edges
    def create_edge_label(row):
        relationship = row.get('relationship', 'connected_to')
        if row.get('efficacy'):
            return f"{relationship} (eff: {row['efficacy']})"
        elif row.get('copay'):
            return f"{relationship} (copay: ${row['copay']})"
        else:
            return relationship
    
    edges_df['edge_label'] = edges_df.apply(create_edge_label, axis=1)
    
    # 5. Create the visualization
    try:
        g = (graphistry
             .bind(source='source', destination='target', node='id')
             .bind(point_title='display_label')
             .bind(point_label='display_label')
             .edges(edges_df)
             .nodes(nodes_df))
        
        # Apply coloring
        if 'type_numeric' in nodes_df.columns:
            g = g.bind(point_color='type_numeric')
        
        # Apply sizing
        if 'prescription_volume' in nodes_df.columns:
            # Normalize prescription volume for better sizing
            max_vol = nodes_df['prescription_volume'].max()
            nodes_df['size_normalized'] = (nodes_df['prescription_volume'] / max_vol) * 10 + 1
            g = g.nodes(nodes_df).bind(point_size='size_normalized')
        
        return g
        
    except Exception as e:
        print(f"Enhanced visualization failed: {e}")
        # Fall back to basic version
        return create_healthcare_graphistry_viz(nodes, edges, graphistry_credentials)

def example_usage():
    """
    Example usage with comprehensive error handling
    """
    
    # Graphistry credentials - REPLACE THESE
    credentials = {
        'username': 'your_username',
        'password': 'your_password'
    }
    
    # Sample healthcare data
    sample_nodes = [
        {
            'id': 'aspirin_325',
            'name': 'Aspirin 325mg',
            'node_type': 'drug',
            'formulary_tier': 1,
            'therapeutic_class': 'NSAID',
            'prescription_volume': 1000,
            'cost': 5.99
        },
        {
            'id': 'ibuprofen_200',
            'name': 'Ibuprofen 200mg', 
            'node_type': 'drug',
            'formulary_tier': 2,
            'therapeutic_class': 'NSAID',
            'prescription_volume': 800,
            'cost': 12.99
        },
        {
            'id': 'headache',
            'name': 'Tension Headache',
            'node_type': 'condition',
            'icd_code': 'G44.209'
        },
        {
            'id': 'bcbs',
            'name': 'Blue Cross Blue Shield',
            'node_type': 'provider',
            'network_size': 'large'
        }
    ]
    
    sample_edges = [
        {
            'source': 'aspirin_325',
            'target': 'headache',
            'relationship': 'treats',
            'efficacy': 0.75
        },
        {
            'source': 'ibuprofen_200',
            'target': 'headache', 
            'relationship': 'treats',
            'efficacy': 0.85
        },
        {
            'source': 'bcbs',
            'target': 'aspirin_325',
            'relationship': 'covers',
            'copay': 5
        },
        {
            'source': 'bcbs',
            'target': 'ibuprofen_200',
            'relationship': 'covers',
            'copay': 15
        }
    ]
    
    print("Creating healthcare knowledge graph visualization...")
    
    # Try enhanced version first
    try:
        print("Attempting enhanced visualization...")
        viz = create_enhanced_healthcare_viz(
            nodes=sample_nodes,
            edges=sample_edges,
            graphistry_credentials=credentials
        )
        
        # Create the plot
        plot_url = viz.plot()
        print(f"✓ Enhanced visualization successful!")
        print(f"View at: {plot_url}")
        return viz
        
    except Exception as e:
        print(f"Enhanced version failed: {e}")
        print("Trying basic visualization...")
        
        try:
            viz = create_healthcare_graphistry_viz(
                nodes=sample_nodes,
                edges=sample_edges,
                graphistry_credentials=credentials
            )
            
            plot_url = viz.plot()
            print(f"✓ Basic visualization successful!")
            print(f"View at: {plot_url}")
            return viz
            
        except Exception as e2:
            print(f"✗ All visualization attempts failed: {e2}")
            print("\nTroubleshooting tips:")
            print("1. Verify your Graphistry credentials are correct")
            print("2. Check your internet connection")
            print("3. Ensure you have a valid Graphistry account/subscription")
            print("4. Try updating the graphistry package: pip install --upgrade graphistry")
            return None

# Simple test function to check Graphistry connection
def test_graphistry_connection(credentials):
    """Test basic Graphistry functionality"""
    try:
        graphistry.register(
            api=3,
            username=credentials['username'],
            password=credentials['password']
        )
        
        # Create minimal test graph
        test_edges = pd.DataFrame({
            'source': ['A', 'B'],
            'target': ['B', 'C']
        })
        
        g = graphistry.bind(source='source', destination='target').edges(test_edges)
        url = g.plot()
        print(f"✓ Graphistry connection test successful: {url}")
        return True
        
    except Exception as e:
        print(f"✗ Graphistry connection test failed: {e}")
        return False

if __name__ == "__main__":
    # Test connection first
    credentials = {
        'username': 'suresh',  # Replace with actual
        'password': 'Bindu@1989'   # Replace with actual
    }
    
    print("Testing Graphistry connection...")
    if test_graphistry_connection(credentials):
        print("\nConnection OK, proceeding with healthcare visualization...")
        example_usage()
    else:
        print("\nPlease fix Graphistry connection issues first.")