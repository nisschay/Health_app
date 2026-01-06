"""
Patient Information Management Module
Handles consolidation and conflict resolution for patient data
"""

import re
import pandas as pd
from typing import Dict, List, Optional
from collections import Counter
from .data_processor import DataProcessor


class PatientInfoManager:
    """
    Manages patient information extraction, normalization, and consolidation.
    Handles conflicts when combining data from multiple sources.
    """
    
    @staticmethod
    def normalize_name(name: str) -> str:
        """Normalize patient name by removing titles and standardizing format"""
        if not name or name == 'N/A':
            return ''
        
        name = name.strip()
        
        # Remove common titles
        titles = ['mr', 'mrs', 'ms', 'dr', 'prof', 'self', 'mr.', 'mrs.', 'ms.', 'dr.', 'prof.']
        name_lower = name.lower()
        
        for title in titles:
            name_lower = re.sub(rf'\b{title}\b\.?\s*', '', name_lower)
        
        words = [word for word in name_lower.split() if word]
        return ' '.join(word.title() for word in words)
    
    @classmethod
    def are_names_matching(cls, name1: str, name2: str) -> bool:
        """Check if two names likely refer to the same person"""
        name1 = cls.normalize_name(name1)
        name2 = cls.normalize_name(name2)
        
        if not name1 or not name2:
            return True  # Empty names don't conflict
        
        name1_parts = set(name1.split())
        name2_parts = set(name2.split())
        
        # Check if one name is subset of other
        if name1_parts.issubset(name2_parts) or name2_parts.issubset(name1_parts):
            return True
        
        # Check overlap
        matching = name1_parts.intersection(name2_parts)
        total = name1_parts.union(name2_parts)
        
        if len(matching) >= 2 and len(matching) / len(total) >= 0.6:
            return True
        
        return False
    
    @classmethod
    def consolidate(cls, patient_info_list: List[Dict]) -> Dict:
        """
        Consolidate patient info from multiple sources.
        Uses smart selection to pick the best value for each field.
        
        Args:
            patient_info_list: List of patient info dictionaries
            
        Returns:
            Consolidated patient info dictionary
        """
        if not patient_info_list:
            return {}
        
        # Filter out empty/invalid entries
        valid_infos = [pi for pi in patient_info_list if pi]
        if not valid_infos:
            return {}
        
        # Extract valid values for each field
        names = cls._get_valid_values(valid_infos, 'name')
        ages = cls._get_valid_values(valid_infos, 'age')
        genders = cls._get_valid_values(valid_infos, 'gender')
        patient_ids = cls._get_valid_values(valid_infos, 'patient_id')
        dates = cls._get_valid_values(valid_infos, 'date')
        lab_names = cls._get_valid_values(valid_infos, 'lab_name')
        
        # Select best name (longest, most complete)
        final_name = cls._select_best_name(names)
        
        # Select age from most recent date
        final_age = cls._select_age_by_date(valid_infos)
        
        # Most common values for other fields
        final_gender = Counter(genders).most_common(1)[0][0] if genders else 'N/A'
        final_id = Counter(patient_ids).most_common(1)[0][0] if patient_ids else 'N/A'
        final_lab = Counter(lab_names).most_common(1)[0][0] if lab_names else 'N/A'
        
        # Most recent date
        final_date = cls._get_most_recent_date(dates)
        
        return {
            'name': final_name,
            'age': final_age,
            'gender': final_gender,
            'patient_id': final_id,
            'date': final_date,
            'lab_name': final_lab
        }
    
    @staticmethod
    def _get_valid_values(info_list: List[Dict], key: str) -> List[str]:
        """Extract non-empty, non-N/A values for a key"""
        values = []
        for info in info_list:
            val = info.get(key)
            if val and val not in ['N/A', '', 'nan', None]:
                values.append(str(val))
        return values
    
    @classmethod
    def _select_best_name(cls, names: List[str]) -> str:
        """Select the best name from a list (longest, most frequent)"""
        if not names:
            return 'N/A'
        
        # Normalize all names
        normalized = [cls.normalize_name(n) for n in names if n]
        normalized = [n for n in normalized if n]  # Remove empty
        
        if not normalized:
            return 'N/A'
        
        # Count occurrences
        counts = Counter(normalized)
        max_count = max(counts.values())
        
        # Among most frequent, pick longest
        most_frequent = [n for n, c in counts.items() if c == max_count]
        return max(most_frequent, key=len)
    
    @classmethod
    def _select_age_by_date(cls, info_list: List[Dict]) -> str:
        """Select age from the entry with most recent date"""
        date_age_pairs = []
        
        for info in info_list:
            date_str = info.get('date', '')
            age = info.get('age', '')
            
            if date_str and date_str not in ['N/A', ''] and age and age not in ['N/A', '']:
                parsed_date = DataProcessor.parse_date(date_str)
                if parsed_date:
                    date_age_pairs.append((parsed_date, age))
        
        if date_age_pairs:
            # Sort by date descending, return age from most recent
            date_age_pairs.sort(key=lambda x: x[0], reverse=True)
            return date_age_pairs[0][1]
        
        # Fallback: most common age
        ages = cls._get_valid_values(info_list, 'age')
        return Counter(ages).most_common(1)[0][0] if ages else 'N/A'
    
    @classmethod
    def _get_most_recent_date(cls, dates: List[str]) -> str:
        """Get the most recent date from a list of date strings"""
        if not dates:
            return 'N/A'
        
        parsed_dates = []
        for d in dates:
            parsed = DataProcessor.parse_date(d)
            if parsed:
                parsed_dates.append(parsed)
        
        if not parsed_dates:
            return 'N/A'
        
        return DataProcessor.format_date(max(parsed_dates))
    
    @classmethod
    def extract_from_dataframe(cls, df: pd.DataFrame) -> Dict:
        """Extract patient info from a DataFrame"""
        if df.empty:
            return {}
        
        info = {}
        
        # Get most common values for each field
        field_mapping = {
            'name': 'Patient_Name',
            'age': 'Age',
            'gender': 'Gender',
            'patient_id': 'Patient_ID',
            'lab_name': 'Lab_Name'
        }
        
        for key, col in field_mapping.items():
            if col in df.columns:
                values = df[col].dropna()
                values = values[values != 'N/A']
                info[key] = values.mode().iloc[0] if not values.empty else 'N/A'
            else:
                info[key] = 'N/A'
        
        # Get most recent date
        if 'Test_Date' in df.columns:
            dates = df['Test_Date'].dropna()
            dates = [d for d in dates if d != 'N/A']
            info['date'] = cls._get_most_recent_date(dates)
        else:
            info['date'] = 'N/A'
        
        return info
    
    @classmethod
    def merge_with_new_data(cls, existing_info: Dict, new_info_list: List[Dict]) -> Dict:
        """
        Merge existing patient info with new data from PDFs.
        New PDF data takes priority for most fields.
        """
        if not new_info_list:
            return existing_info or {}
        
        if not existing_info:
            return cls.consolidate(new_info_list)
        
        # Consolidate new info
        new_consolidated = cls.consolidate(new_info_list)
        
        # Merge: prefer new data over existing
        merged = {}
        
        for key in ['name', 'age', 'gender', 'patient_id', 'lab_name', 'date']:
            new_val = new_consolidated.get(key)
            existing_val = existing_info.get(key)
            
            # Prefer new value if it's valid
            if new_val and new_val not in ['N/A', '', 'nan']:
                merged[key] = new_val
            elif existing_val and existing_val not in ['N/A', '', 'nan']:
                merged[key] = existing_val
            else:
                merged[key] = 'N/A'
        
        return merged
