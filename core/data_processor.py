"""
Data Processing Module
Handles DataFrame creation, normalization, and transformations
"""

import re
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter


class DataProcessor:
    """
    Centralized data processing with optimized transformations.
    Handles conversion from AI results to structured DataFrames.
    """
    
    # Standard column order
    COLUMNS = [
        'Source_Filename', 'Patient_ID', 'Patient_Name', 'Age', 'Gender',
        'Test_Date', 'Lab_Name', 'Test_Category', 'Original_Test_Name',
        'Test_Name', 'Result', 'Unit', 'Reference_Range', 'Status',
        'Processed_Date', 'Result_Numeric', 'Test_Date_dt'
    ]
    
    @staticmethod
    def parse_date(date_str: str) -> Optional[datetime]:
        """Parse date string in DD-MM-YYYY format (day first)"""
        if not date_str or pd.isna(date_str) or date_str in ['N/A', '']:
            return None
        
        date_str = str(date_str).strip()
        
        # Try common formats (day first)
        formats = ['%d/%m/%Y', '%d-%m-%Y', '%d.%m.%Y']
        
        for fmt in formats:
            try:
                return pd.to_datetime(date_str, format=fmt)
            except:
                continue
        
        # Fallback with dayfirst=True
        try:
            return pd.to_datetime(date_str, dayfirst=True)
        except:
            return None
    
    @staticmethod
    def format_date(date_obj) -> str:
        """Format datetime to DD-MM-YYYY string"""
        if pd.isna(date_obj) or date_obj is None:
            return 'N/A'
        return date_obj.strftime('%d-%m-%Y')
    
    @classmethod
    def create_dataframe_from_ai_result(cls, ai_result: Dict, 
                                         source_filename: str = "Uploaded PDF") -> Tuple[pd.DataFrame, Dict]:
        """
        Convert AI extraction result to structured DataFrame.
        
        Returns:
            Tuple of (DataFrame, patient_info dict)
        """
        if not ai_result or 'test_results' not in ai_result:
            return pd.DataFrame(), {}
        
        patient_info = ai_result.get('patient_info', {})
        
        # Parse and format date
        date_str = patient_info.get('date', 'N/A')
        parsed_date = cls.parse_date(date_str)
        formatted_date = cls.format_date(parsed_date) if parsed_date else 'N/A'
        patient_info['date'] = formatted_date
        
        rows = []
        for test in ai_result.get('test_results', []):
            row = {
                'Source_Filename': source_filename,
                'Patient_ID': patient_info.get('patient_id', 'N/A'),
                'Patient_Name': patient_info.get('name', 'N/A'),
                'Age': patient_info.get('age', 'N/A'),
                'Gender': patient_info.get('gender', 'N/A'),
                'Test_Date': formatted_date,
                'Lab_Name': patient_info.get('lab_name', 'N/A'),
                'Test_Category': cls._clean_value(test.get('category', 'Other')),
                'Original_Test_Name': test.get('test_name', 'UnknownTest'),
                'Test_Name': cls._clean_value(test.get('test_name', 'UnknownTest')),
                'Result': test.get('result', ''),
                'Unit': test.get('unit', ''),
                'Reference_Range': test.get('reference_range', ''),
                'Status': cls._clean_value(test.get('status', 'N/A')),
                'Processed_Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            }
            rows.append(row)
        
        if not rows:
            return pd.DataFrame(), patient_info
        
        df = pd.DataFrame(rows)
        
        # Add computed columns
        df['Result_Numeric'] = pd.to_numeric(df['Result'], errors='coerce')
        df['Test_Date_dt'] = df['Test_Date'].apply(cls.parse_date)
        
        # Sort by date and category
        df = df.sort_values(
            by=['Test_Date_dt', 'Test_Category', 'Test_Name'],
            na_position='last'
        ).reset_index(drop=True)
        
        return df, patient_info
    
    @staticmethod
    def _clean_value(value: str, default_case: str = 'title') -> str:
        """Clean and normalize string values"""
        if not isinstance(value, str):
            return str(value) if value else 'N/A'
        
        value = value.strip()
        if not value:
            return 'N/A'
        
        if default_case == 'title':
            return value.title()
        elif default_case == 'lower':
            return value.lower()
        return value
    
    @classmethod
    def combine_dataframes(cls, dfs: List[pd.DataFrame]) -> pd.DataFrame:
        """Combine multiple DataFrames with consistent columns"""
        if not dfs:
            return pd.DataFrame()
        
        # Filter out empty DataFrames
        valid_dfs = [df for df in dfs if not df.empty]
        
        if not valid_dfs:
            return pd.DataFrame()
        
        # Ensure all DataFrames have required columns
        for df in valid_dfs:
            for col in cls.COLUMNS:
                if col not in df.columns:
                    df[col] = 'N/A'
        
        # Combine and reorder columns
        combined = pd.concat(valid_dfs, ignore_index=True)
        
        # Keep only defined columns plus any extras
        existing_cols = [c for c in cls.COLUMNS if c in combined.columns]
        combined = combined[existing_cols]
        
        return combined
    
    @classmethod
    def remove_duplicates(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate test entries, keeping the most informative one"""
        if df.empty:
            return df
        
        df = df.copy()
        
        # Group by test name and date, keep the row with most data
        def get_data_completeness(row):
            """Score row by how complete its data is"""
            score = 0
            for col in ['Result', 'Reference_Range', 'Status', 'Unit']:
                if col in row.index and row[col] and str(row[col]) not in ['', 'N/A', 'nan']:
                    score += 1
            return score
        
        df['_completeness'] = df.apply(get_data_completeness, axis=1)
        df = df.sort_values('_completeness', ascending=False)
        df = df.drop_duplicates(subset=['Test_Name', 'Test_Date'], keep='first')
        df = df.drop(columns=['_completeness'])
        
        return df.reset_index(drop=True)
    
    @classmethod  
    def unify_test_names(cls, df: pd.DataFrame, threshold: int = 90) -> pd.DataFrame:
        """
        Unify similar test names using fuzzy matching.
        This is a simplified version - the full implementation is in unify_test_names.py
        """
        if df.empty:
            return df
        
        # Handle common vitamin B12 variants (most common issue)
        b12_variants = [
            'Vitamin B-12', 'Vitamin B12', 'Vitamin B 12',
            'vitamin b-12', 'vitamin b12', 'vitamin b 12',
            'Vitamin B-12 Level', 'Vitamin B12 Level'
        ]
        
        df = df.copy()
        df['Test_Name'] = df['Test_Name'].fillna('N/A')
        
        for variant in b12_variants:
            df.loc[df['Test_Name'].str.lower() == variant.lower(), 'Test_Name'] = 'Vitamin B12'
        
        return df
    
    @staticmethod
    def parse_reference_range(ref_str: str) -> Tuple[Optional[float], Optional[float], Optional[str]]:
        """
        Parse reference range string.
        
        Returns:
            Tuple of (low, high, type) where type is 'range', 'less_than', 'greater_than', or None
        """
        if not isinstance(ref_str, str) or ref_str.lower() in ['n/a', ''] or not ref_str.strip():
            return None, None, None
        
        ref_str = ref_str.strip()
        
        # Try range format (e.g., "13.0 - 17.0")
        match = re.search(r'([\d.]+)\s*-\s*([\d.]+)', ref_str)
        if match:
            try:
                return float(match.group(1)), float(match.group(2)), "range"
            except ValueError:
                pass
        
        # Try less than (e.g., "< 5.0")
        match = re.search(r'(?:<|Less than|upto)\s*([\d.]+)', ref_str, re.IGNORECASE)
        if match:
            try:
                return None, float(match.group(1)), "less_than"
            except ValueError:
                pass
        
        # Try greater than (e.g., "> 5.0")
        match = re.search(r'(?:>|Greater than|above)\s*([\d.]+)', ref_str, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1)), None, "greater_than"
            except ValueError:
                pass
        
        return None, None, None
