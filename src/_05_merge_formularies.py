import pandas as pd
from pathlib import Path
from rapidfuzz import process, fuzz

def merge_formularies(base_path: Path, output_filename: str = "merged_drug_details.xlsx",
                      threshold: float = 90.0) -> pd.DataFrame:
    """
    Merge drug details from multiple formulary Excel files using fuzzy matching.
    
    Parameters:
        base_path (Path): Directory containing the Excel files.
        output_filename (str): Name of the output Excel file.
        threshold (float): Minimum fuzzy match score to consider two names the same.
    
    Returns:
        pd.DataFrame: DataFrame with unique drug details and a single 'Formulary' field.
    """
    # List of files to process (Optum is primary)
    file_order = ['Optum.xlsx', 'Aetna.xlsx', 'Anthem.xlsx', 'Cigna.xlsx', 'Humana.xlsx']
    # Column names
    columns = ['Drug Family', 'Drug Name', 'Drug Tier', 'Notes', 'Formulary']
    
    # Load primary dataset (Optum) and use it as initial aggregated list
    optum_df = pd.read_excel(base_path / 'Optum.xlsx')[columns]
    agg_records = optum_df.to_dict('records')
    agg_names = [record['Drug Name'] for record in agg_records]
    
    # Process other datasets and add unique entries
    for filename in file_order[1:]:
        df = pd.read_excel(base_path / filename)[columns]
        for record in df.to_dict('records'):
            name = record['Drug Name']
            if isinstance(name, str) and name.strip():
                # Fuzzy match against existing names in the aggregated list
                match = process.extractOne(
                    name,
                    agg_names,
                    scorer=fuzz.token_set_ratio,
                    processor=str.lower,
                    score_cutoff=threshold
                )
                if match is None:
                    # No match above threshold: add new record
                    agg_records.append(record)
                    agg_names.append(name)
    
    # Create final DataFrame and save to Excel
    final_df = pd.DataFrame(agg_records)
    final_df.to_excel(base_path / output_filename, index=False)
    return final_df

# Example usage:
if __name__ == "__main__":
    base_dir = Path("data/output")  # adjust if your files are elsewhere
    merged_df = merge_formularies(base_dir, "merged_drug_details.xlsx", threshold=90.0)
    print(f"Merged file saved with {len(merged_df)} unique drugs.")
