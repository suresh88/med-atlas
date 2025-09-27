### Core Libraries:
import pandas as pd
import numpy as np
import re
import requests
import json
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# For advanced name matching and validation
import difflib
from fuzzywuzzy import fuzz, process
import nltk
from nltk.corpus import stopwords

# For API-based validation (optional)
import time
from functools import lru_cache
import concurrent.futures
from threading import Lock

# For progress tracking
from tqdm import tqdm
tqdm.pandas()

### Core Extraction Class:

class DrugNameExtractor:
    """
    Efficient drug name extractor for separating generic and brand names
    from pharmaceutical databases with high accuracy and performance.
    """
    
    def __init__(self, 
                 use_external_api: bool = False,
                 confidence_threshold: float = 0.8,
                 batch_size: int = 100):
        """
        Initialize the drug name extractor
        
        Args:
            use_external_api: Whether to use external APIs for validation
            confidence_threshold: Minimum confidence score for extractions
            batch_size: Batch size for processing large datasets
        """
        self.use_external_api = use_external_api
        self.confidence_threshold = confidence_threshold
        self.batch_size = batch_size
        self.extraction_patterns = self._compile_extraction_patterns()
        self.known_generics = self._load_known_generic_names()
        self.known_brands = self._load_known_brand_names()
        self.extraction_stats = {
            'total_processed': 0,
            'successful_extractions': 0,
            'generic_only': 0,
            'brand_only': 0,
            'both_extracted': 0,
            'low_confidence': 0
        }
    
    def _compile_extraction_patterns(self) -> List[Dict]:
        """Compile regex patterns for different drug name formats"""
        
        patterns = [
            # Pattern 1: Generic (Brand) or Brand (Generic)
            {
                'pattern': r'^([^(]+)\s*\(([^)]+)\)$',
                'description': 'Parentheses format',
                'confidence': 0.9
            },
            
            # Pattern 2: Generic / Brand or Brand / Generic
            {
                'pattern': r'^([^/]+)\s*/\s*([^/]+)$',
                'description': 'Slash separator',
                'confidence': 0.85
            },
            
            # Pattern 3: Generic - Brand or Brand - Generic
            {
                'pattern': r'^([^-]+)\s*-\s*([^-]+)$',
                'description': 'Dash separator',
                'confidence': 0.8
            },
            
            # Pattern 4: Multiple formats with keywords
            {
                'pattern': r'^(.+?)\s+(?:brand|generic|trade)\s*[:=]\s*(.+)$',
                'description': 'Keyword separator',
                'confidence': 0.85
            },
            
            # Pattern 5: Comma-separated with brand indicators
            {
                'pattern': r'^([^,]+),\s*(?:brand|trademark|®|™)\s*([^,]+)$',
                'description': 'Comma with brand indicators',
                'confidence': 0.9
            }
        ]
        
        # Compile regex patterns for efficiency
        for pattern in patterns:
            pattern['compiled'] = re.compile(pattern['pattern'], re.IGNORECASE)
        
        return patterns
    
    def _load_known_generic_names(self) -> set:
        """Load known generic drug names for validation"""
        
        # Common generic drug names for validation
        known_generics = {
            'acetaminophen', 'ibuprofen', 'aspirin', 'metformin', 'lisinopril',
            'amlodipine', 'metoprolol', 'omeprazole', 'simvastatin', 'atorvastatin',
            'levothyroxine', 'azithromycin', 'amoxicillin', 'hydrochlorothiazide',
            'losartan', 'albuterol', 'gabapentin', 'sertraline', 'trazodone',
            'prednisone', 'tramadol', 'cyclobenzaprine', 'ciprofloxacin'
            # Add more as needed from pharmaceutical databases
        }
        
        return known_generics
    
    def _load_known_brand_names(self) -> set:
        """Load known brand drug names for validation"""
        
        # Common brand drug names for validation
        known_brands = {
            'tylenol', 'advil', 'motrin', 'glucophage', 'prinivil', 'zestril',
            'norvasc', 'lopressor', 'toprol', 'prilosec', 'zocor', 'lipitor',
            'synthroid', 'zithromax', 'amoxil', 'microzide', 'cozaar',
            'proventil', 'neurontin', 'zoloft', 'desyrel', 'deltasone'
            # Add more as needed from pharmaceutical databases
        }
        
        return known_brands
    
    def extract_drug_names(self, drug_name: str) -> Dict[str, Union[str, float]]:
        """
        Extract generic and brand names from a single drug name entry
        
        Args:
            drug_name: Raw drug name string
            
        Returns:
            Dictionary with extracted names and confidence scores
        """
        
        if pd.isna(drug_name) or not isinstance(drug_name, str):
            return {
                'generic_name': None,
                'brand_name': None,
                'confidence': 0.0,
                'extraction_method': 'invalid_input'
            }
        
        # Clean the input
        cleaned_name = self._clean_drug_name(drug_name)
        
        # Try pattern-based extraction
        result = self._pattern_based_extraction(cleaned_name)
        
        if result['confidence'] >= self.confidence_threshold:
            return result
        
        # Try knowledge-based extraction
        kb_result = self._knowledge_based_extraction(cleaned_name)
        
        if kb_result['confidence'] >= self.confidence_threshold:
            return kb_result
        
        # Try fuzzy matching approach
        fuzzy_result = self._fuzzy_matching_extraction(cleaned_name)
        
        # Return the best result
        best_result = max([result, kb_result, fuzzy_result], 
                         key=lambda x: x['confidence'])
        
        return best_result
    
    def _clean_drug_name(self, drug_name: str) -> str:
        """Clean and normalize drug name for processing"""
        
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', drug_name.strip())
        
        # Remove common suffixes that don't affect name extraction
        suffixes_to_remove = [
            r'\s+\d+\s*mg\b', r'\s+\d+\s*mcg\b', r'\s+\d+\s*g\b',
            r'\s+tablets?\b', r'\s+capsules?\b', r'\s+injection\b',
            r'\s+oral\b', r'\s+topical\b', r'\s+cream\b', r'\s+gel\b'
        ]
        
        for suffix in suffixes_to_remove:
            cleaned = re.sub(suffix, '', cleaned, flags=re.IGNORECASE)
        
        # Remove trademark symbols
        cleaned = re.sub(r'[®™]', '', cleaned)
        
        return cleaned.strip()
    
    def _pattern_based_extraction(self, drug_name: str) -> Dict[str, Union[str, float]]:
        """Extract names using regex patterns"""
        
        for pattern_info in self.extraction_patterns:
            match = pattern_info['compiled'].match(drug_name)
            
            if match:
                part1, part2 = match.groups()
                part1, part2 = part1.strip(), part2.strip()
                
                # Determine which is generic and which is brand
                generic, brand, confidence = self._classify_parts(
                    part1, part2, pattern_info['confidence']
                )
                
                return {
                    'generic_name': generic,
                    'brand_name': brand,
                    'confidence': confidence,
                    'extraction_method': f"pattern_{pattern_info['description']}"
                }
        
        return {
            'generic_name': None,
            'brand_name': None,
            'confidence': 0.0,
            'extraction_method': 'no_pattern_match'
        }
    
    def _classify_parts(self, part1: str, part2: str, base_confidence: float) -> Tuple[str, str, float]:
        """Classify which part is generic and which is brand"""
        
        # Check against known lists
        part1_lower = part1.lower()
        part2_lower = part2.lower()
        
        part1_is_generic = part1_lower in self.known_generics
        part1_is_brand = part1_lower in self.known_brands
        part2_is_generic = part2_lower in self.known_generics
        part2_is_brand = part2_lower in self.known_brands
        
        # Direct matches
        if part1_is_generic and part2_is_brand:
            return part1, part2, base_confidence * 1.1
        elif part1_is_brand and part2_is_generic:
            return part2, part1, base_confidence * 1.1
        
        # Heuristic-based classification
        # Generic names are typically longer and more descriptive
        # Brand names are often shorter and more memorable
        
        confidence_adjustment = 0.0
        
        # Length-based heuristic
        if len(part1) > len(part2) * 1.5:
            generic, brand = part1, part2
            confidence_adjustment += 0.1
        elif len(part2) > len(part1) * 1.5:
            generic, brand = part2, part1
            confidence_adjustment += 0.1
        else:
            # Default: assume first part is generic
            generic, brand = part1, part2
        
        # Chemical naming patterns (generic drugs often have chemical-sounding names)
        chemical_patterns = [r'[a-z]+ine$', r'[a-z]+ol$', r'[a-z]+ate$', r'[a-z]+ide$']
        
        for pattern in chemical_patterns:
            if re.search(pattern, part1.lower()):
                generic, brand = part1, part2
                confidence_adjustment += 0.05
                break
            elif re.search(pattern, part2.lower()):
                generic, brand = part2, part1
                confidence_adjustment += 0.05
                break
        
        final_confidence = min(base_confidence + confidence_adjustment, 1.0)
        return generic, brand, final_confidence
    
    def _knowledge_based_extraction(self, drug_name: str) -> Dict[str, Union[str, float]]:
        """Extract using pharmaceutical knowledge base"""
        
        name_lower = drug_name.lower()
        
        # Check if the entire name is a known generic or brand
        if name_lower in self.known_generics:
            return {
                'generic_name': drug_name,
                'brand_name': None,
                'confidence': 0.9,
                'extraction_method': 'known_generic'
            }
        elif name_lower in self.known_brands:
            return {
                'generic_name': None,
                'brand_name': drug_name,
                'confidence': 0.9,
                'extraction_method': 'known_brand'
            }
        
        # Check for partial matches
        for known_generic in self.known_generics:
            if known_generic in name_lower:
                remaining = drug_name.replace(known_generic, '').strip(' -/,()')
                if remaining:
                    return {
                        'generic_name': known_generic.title(),
                        'brand_name': remaining,
                        'confidence': 0.8,
                        'extraction_method': 'partial_generic_match'
                    }
        
        return {
            'generic_name': None,
            'brand_name': None,
            'confidence': 0.0,
            'extraction_method': 'no_knowledge_match'
        }
    
    def _fuzzy_matching_extraction(self, drug_name: str) -> Dict[str, Union[str, float]]:
        """Extract using fuzzy string matching"""
        
        # Find closest matches in known databases
        generic_matches = process.extract(
            drug_name.lower(), 
            self.known_generics, 
            limit=3, 
            scorer=fuzz.ratio
        )
        
        brand_matches = process.extract(
            drug_name.lower(), 
            self.known_brands, 
            limit=3, 
            scorer=fuzz.ratio
        )
        
        best_generic = generic_matches[0] if generic_matches and generic_matches[0][1] > 70 else None
        best_brand = brand_matches[0] if brand_matches and brand_matches[0][1] > 70 else None
        
        if best_generic and best_brand:
            # Choose the better match
            if best_generic[1] > best_brand[1]:
                return {
                    'generic_name': best_generic[0].title(),
                    'brand_name': None,
                    'confidence': best_generic[1] / 100.0 * 0.8,
                    'extraction_method': 'fuzzy_generic_match'
                }
            else:
                return {
                    'generic_name': None,
                    'brand_name': best_brand[0].title(),
                    'confidence': best_brand[1] / 100.0 * 0.8,
                    'extraction_method': 'fuzzy_brand_match'
                }
        elif best_generic:
            return {
                'generic_name': best_generic[0].title(),
                'brand_name': None,
                'confidence': best_generic[1] / 100.0 * 0.8,
                'extraction_method': 'fuzzy_generic_match'
            }
        elif best_brand:
            return {
                'generic_name': None,
                'brand_name': best_brand[0].title(),
                'confidence': best_brand[1] / 100.0 * 0.8,
                'extraction_method': 'fuzzy_brand_match'
            }
        
        return {
            'generic_name': None,
            'brand_name': None,
            'confidence': 0.0,
            'extraction_method': 'no_fuzzy_match'
        }
    
    def process_dataframe(self, df: pd.DataFrame, drug_name_column: str = 'Drug Name') -> pd.DataFrame:
        """
        Process entire dataframe to extract drug names
        
        Args:
            df: Input DataFrame
            drug_name_column: Name of the column containing drug names
            
        Returns:
            DataFrame with added Generic Name and Brand Name columns
        """
        
        if drug_name_column not in df.columns:
            raise ValueError(f"Column '{drug_name_column}' not found in DataFrame")
        
        print(f"Processing {len(df)} drug names...")
        
        # Apply extraction function with progress bar
        extraction_results = df[drug_name_column].progress_apply(self.extract_drug_names)
        
        # Convert results to separate columns
        df['Generic Name'] = extraction_results.apply(lambda x: x['generic_name'])
        df['Brand Name'] = extraction_results.apply(lambda x: x['brand_name'])
        df['Extraction Confidence'] = extraction_results.apply(lambda x: x['confidence'])
        df['Extraction Method'] = extraction_results.apply(lambda x: x['extraction_method'])
        
        # Update statistics
        self._update_extraction_stats(extraction_results)
        
        return df
    
    def _update_extraction_stats(self, results: pd.Series):
        """Update extraction statistics"""
        
        self.extraction_stats['total_processed'] = len(results)
        
        for result in results:
            if result['generic_name'] and result['brand_name']:
                self.extraction_stats['both_extracted'] += 1
            elif result['generic_name']:
                self.extraction_stats['generic_only'] += 1
            elif result['brand_name']:
                self.extraction_stats['brand_only'] += 1
            
            if result['confidence'] >= self.confidence_threshold:
                self.extraction_stats['successful_extractions'] += 1
            else:
                self.extraction_stats['low_confidence'] += 1
    
    def get_extraction_report(self) -> Dict:
        """Generate detailed extraction report"""
        
        total = self.extraction_stats['total_processed']
        
        if total == 0:
            return {"message": "No data processed yet"}
        
        return {
            'summary': {
                'total_processed': total,
                'success_rate': f"{(self.extraction_stats['successful_extractions'] / total * 100):.1f}%",
                'both_names_extracted': f"{(self.extraction_stats['both_extracted'] / total * 100):.1f}%",
                'generic_only_extracted': f"{(self.extraction_stats['generic_only'] / total * 100):.1f}%",
                'brand_only_extracted': f"{(self.extraction_stats['brand_only'] / total * 100):.1f}%",
                'low_confidence_extractions': f"{(self.extraction_stats['low_confidence'] / total * 100):.1f}%"
            },
            'recommendations': self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on extraction statistics"""
        
        recommendations = []
        stats = self.extraction_stats
        total = stats['total_processed']
        
        if total == 0:
            return recommendations
        
        low_confidence_rate = stats['low_confidence'] / total
        if low_confidence_rate > 0.2:
            recommendations.append("Consider manual review of low-confidence extractions")
        
        both_extracted_rate = stats['both_extracted'] / total
        if both_extracted_rate < 0.3:
            recommendations.append("Many entries contain only one name type - verify data source")
        
        if stats['generic_only'] > stats['brand_only'] * 2:
            recommendations.append("Dataset appears to be generic-name focused")
        elif stats['brand_only'] > stats['generic_only'] * 2:
            recommendations.append("Dataset appears to be brand-name focused")
        
        return recommendations

# Main execution function
def extract_drug_names_from_excel(file_path: str, 
                                 drug_name_column: str = 'Drug Name',
                                 output_file: Optional[str] = None) -> pd.DataFrame:
    """
    Main function to extract drug names from Excel file
    
    Args:
        file_path: Path to input Excel file
        drug_name_column: Name of the column containing drug names
        output_file: Optional output file path
        
    Returns:
        DataFrame with extracted drug names
    """
    
    try:
        # Load the Excel file
        print(f"Loading Excel file: {file_path}")
        df = pd.read_excel(file_path)
        
        print(f"Loaded {len(df)} rows")
        print(f"Columns: {list(df.columns)}")
        
        if drug_name_column not in df.columns:
            print(f"Available columns: {list(df.columns)}")
            raise ValueError(f"Column '{drug_name_column}' not found in the Excel file")
        
        # Initialize extractor
        extractor = DrugNameExtractor(
            use_external_api=False,
            confidence_threshold=0.7,
            batch_size=100
        )
        
        # Process the dataframe
        processed_df = extractor.process_dataframe(df, drug_name_column)
        
        # Generate and display report
        report = extractor.get_extraction_report()
        
        print("\n" + "="*50)
        print("EXTRACTION REPORT")
        print("="*50)
        
        for key, value in report['summary'].items():
            print(f"{key.replace('_', ' ').title()}: {value}")
        
        if report.get('recommendations'):
            print("\nRecommendations:")
            for rec in report['recommendations']:
                print(f"- {rec}")
        
        # Save results if output file specified
        if output_file:
            processed_df.to_excel(output_file, index=False)
            print(f"\nResults saved to: {output_file}")
        
        # Display sample results
        print("\n" + "="*50)
        print("SAMPLE RESULTS")
        print("="*50)
        
        sample_cols = [drug_name_column, 'Generic Name', 'Brand Name', 'Extraction Confidence', 'Extraction Method']
        sample_df = processed_df[sample_cols].head(10)
        
        for _, row in sample_df.iterrows():
            print(f"\nOriginal: {row[drug_name_column]}")
            print(f"Generic: {row['Generic Name'] if pd.notna(row['Generic Name']) else 'Not found'}")
            print(f"Brand: {row['Brand Name'] if pd.notna(row['Brand Name']) else 'Not found'}")
            print(f"Confidence: {row['Extraction Confidence']:.2f}")
            print(f"Method: {row['Extraction Method']}")
        
        return processed_df
        
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        raise

# Usage example and testing
if __name__ == "__main__":
    # Example usage
    file_path = "data/output/Optum.xlsx"  # Replace with your file path
    
    # Process the file
    result_df = extract_drug_names_from_excel(
        file_path=file_path,
        drug_name_column="Drug Name",  # Adjust column name if different
        output_file="data/output/Optum_extracted.xlsx"
    )
    
    # Additional analysis
    print("\n" + "="*50)
    print("ADDITIONAL STATISTICS")
    print("="*50)
    
    # Confidence distribution
    confidence_dist = result_df['Extraction Confidence'].describe()
    print("\nConfidence Score Distribution:")
    print(confidence_dist)
    
    # Method usage
    method_counts = result_df['Extraction Method'].value_counts()
    print("\nExtraction Methods Used:")
    print(method_counts)
    
    # Quality assessment
    high_confidence = (result_df['Extraction Confidence'] >= 0.8).sum()
    medium_confidence = ((result_df['Extraction Confidence'] >= 0.5) & 
                        (result_df['Extraction Confidence'] < 0.8)).sum()
    low_confidence = (result_df['Extraction Confidence'] < 0.5).sum()
    
    print(f"\nQuality Assessment:")
    print(f"High Confidence (≥0.8): {high_confidence} ({high_confidence/len(result_df)*100:.1f}%)")
    print(f"Medium Confidence (0.5-0.8): {medium_confidence} ({medium_confidence/len(result_df)*100:.1f}%)")
    print(f"Low Confidence (<0.5): {low_confidence} ({low_confidence/len(result_df)*100:.1f}%)")