#!/usr/bin/env python3
"""
Excel Dictionary Keys Extractor
Extracts keys from dictionary lists in Excel columns and creates new columns with these keys.
"""

import pandas as pd
import ast
import argparse
import sys
from pathlib import Path


def extract_unique_diseases(cell_value):
    """
    Safely extract keys from a cell containing a list of dictionaries.
    
    Args:
        cell_value: Cell value that should contain a string representation of a list of dictionaries
        
    Returns:
        list: List of keys extracted from the dictionaries, or empty list if extraction fails
    """
    if pd.isna(cell_value) or cell_value == '':
        return []
    
    try:
        # Handle string representation of list
        if isinstance(cell_value, str):
            # Use ast.literal_eval for safe evaluation of string literals
            parsed_data = ast.literal_eval(cell_value)
        else:
            # If it's already a list, use it directly
            parsed_data = cell_value
        
        # Extract keys from list of dictionaries
        if isinstance(parsed_data, list):
            keys = []
            for item in parsed_data:
                if isinstance(item, dict):
                    keys.extend(item.keys())
            return keys
        else:
            return []
            
    except (ValueError, SyntaxError, TypeError) as e:
        print(f"Warning: Could not parse cell value: {cell_value[:100]}... Error: {e}")
        return []


def process_excel_file(input_file, output_file):
    """
    Process the Excel file to extract dictionary keys and create new columns.
    
    Args:
        input_file (str): Path to input Excel file
        output_file (str): Path to output Excel file
    """
    try:
        # Read the Excel file
        print(f"Reading Excel file: {input_file}")
        df = pd.read_excel(input_file)
        
        # Define the columns to process
        columns_to_process = ['may_prevent', 'ci_with', 'may_diagnose', 'may_treat']
        
        # Check if all required columns exist
        missing_columns = [col for col in columns_to_process if col not in df.columns]
        if missing_columns:
            print(f"Warning: Missing columns in the input file: {missing_columns}")
            columns_to_process = [col for col in columns_to_process if col in df.columns]
        
        # Process each column
        for col in columns_to_process:
            print(f"Processing column: {col}")
            
            # Create new column name for extracted keys
            new_col_name = f"{col}_diseases"
            
            # Extract keys from each cell in the column
            df[new_col_name] = df[col].apply(extract_unique_diseases)
            
            print(f"Created new column: {new_col_name}")
        
        # Save the processed DataFrame to a new Excel file
        print(f"Saving processed data to: {output_file}")
        df.to_excel(output_file, index=False)
        
        # Print summary
        print("\nProcessing Summary:")
        print(f"Total rows processed: {len(df)}")
        print(f"Original columns: {list(df.columns)}")
        print(f"New columns created: {[f'{col}_diseases' for col in columns_to_process]}")
        
        # Show sample of extracted keys for verification
        print("\nSample of extracted keys:")
        for col in columns_to_process:
            if col in df.columns:
                new_col = f"{col}_diseases"
                sample_diseases = df[new_col].iloc[0] if len(df) > 0 else []
                print(f"{new_col}: {sample_diseases}")
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing file: {e}")
        sys.exit(1)


def main():
    """Main function to handle command line arguments and execute processing."""
    parser = argparse.ArgumentParser(
        description="Extract keys from dictionary lists in Excel columns",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python script.py input.xlsx output.xlsx
  python script.py data/medical_data.xlsx results/processed_data.xlsx
        """
    )
    
    parser.add_argument(
        '--input_file',
        help='Path to the input Excel file (.xlsx)'
    )
    
    parser.add_argument(
        '--output_file',
        help='Path to the output Excel file (.xlsx)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Validate input file
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file '{args.input_file}' does not exist.")
        sys.exit(1)
    
    if not input_path.suffix.lower() == '.xlsx':
        print("Warning: Input file does not have .xlsx extension.")
    
    # Ensure output directory exists
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Process the file
    process_excel_file(args.input_file, args.output_file)
    print("\nProcessing completed successfully!")


if __name__ == "__main__":
    main()