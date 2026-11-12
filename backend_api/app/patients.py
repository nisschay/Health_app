"""Merge patient details across reports, and read a previously exported CSV or Excel file."""
from __future__ import annotations

import logging
import re
from collections import Counter
from datetime import datetime

import pandas as pd

from .values import format_date_dd_mm_yyyy, parse_date_dd_mm_yyyy, parse_result_numeric

logger = logging.getLogger(__name__)


def extract_patient_info_from_excel_filename(filename):
    """Try to extract patient info from Excel filename if possible"""
    # This is a fallback - try to extract patient name from filename
    name_match = re.search(r'medical_reports?_.*?_([^_]+(?:_[^_]+)*)', filename.lower())
    if name_match:
        # Replace underscores with spaces and title case
        extracted_name = name_match.group(1).replace('_', ' ').title()
        return extracted_name
    return "N/A"


def create_consolidated_info_with_smart_selection(patient_info_list):
    """Create consolidated patient info with smart selection for name and age."""
    if not patient_info_list:
        return {}

    # Get all valid data
    names = [pi.get('name') for pi in patient_info_list if pi.get('name') and pi.get('name') not in ['N/A', '']]
    ages = [pi.get('age') for pi in patient_info_list if pi.get('age') and pi.get('age') not in ['N/A', '']]
    genders = [pi.get('gender') for pi in patient_info_list if pi.get('gender') and pi.get('gender') not in ['N/A', '']]
    patient_ids = [pi.get('patient_id') for pi in patient_info_list if pi.get('patient_id') and pi.get('patient_id') not in ['N/A', '']]
    dates = [pi.get('date') for pi in patient_info_list if pi.get('date') and pi.get('date') not in ['N/A', '']]
    lab_names = [pi.get('lab_name') for pi in patient_info_list if pi.get('lab_name') and pi.get('lab_name') not in ['N/A', '']]

    # 1. Smart name selection - prefer the longest, most complete name
    final_name = "N/A"
    if names:
        # Normalize names first
        normalized_names = [normalize_name(name) for name in names]
        normalized_names = [name for name in normalized_names if name]  # Remove empty strings

        if normalized_names:
            # Count occurrences of normalized names
            name_counts = Counter(normalized_names)

            # Get the most frequent names
            max_count = max(name_counts.values())
            most_frequent_names = [name for name, count in name_counts.items() if count == max_count]

            # Among the most frequent names, choose the longest one (most complete)
            final_name = max(most_frequent_names, key=len) if most_frequent_names else normalized_names[0]

    # 2. Age from the PDF with the most recent date
    final_age = "N/A"
    if dates and ages:
        # Create list of (date, patient_info) pairs
        date_info_pairs = []
        for pi in patient_info_list:
            pi_date = pi.get('date')
            pi_age = pi.get('age')
            if pi_date and pi_date not in ['N/A', ''] and pi_age and pi_age not in ['N/A', '']:
                parsed_date = parse_date_dd_mm_yyyy(pi_date)
                if parsed_date is not None:
                    date_info_pairs.append((parsed_date, pi_age))

        if date_info_pairs:
            # Sort by date and get the age from the most recent date
            date_info_pairs.sort(key=lambda x: x[0], reverse=True)
            final_age = date_info_pairs[0][1]  # Age from most recent date
        elif ages:
            # If no valid dates, just use the most common age
            final_age = Counter(ages).most_common(1)[0][0]

    # 3. Other fields: most frequent valid value
    final_gender = Counter(genders).most_common(1)[0][0] if genders else "N/A"
    final_patient_id = Counter(patient_ids).most_common(1)[0][0] if patient_ids else "N/A"
    final_lab_name = Counter(lab_names).most_common(1)[0][0] if lab_names else "N/A"

    # For date, use the most recent one
    parsed_dates = [parse_date_dd_mm_yyyy(d) for d in dates]
    valid_parsed_dates = [d for d in parsed_dates if d is not None]
    final_date = format_date_dd_mm_yyyy(max(valid_parsed_dates)) if valid_parsed_dates else "N/A"

    return {
        'name': final_name,
        'age': final_age,
        'gender': final_gender,
        'patient_id': final_patient_id,
        'date': final_date,
        'lab_name': final_lab_name
    }


def normalize_name(name):
    """Normalize name by removing titles and standardizing format"""
    if not name or name == 'N/A':
        return ''
    name = name.strip()
    titles = ['mr', 'mrs', 'ms', 'dr', 'prof', 'self', 'mr.', 'mrs.', 'ms.', 'dr.', 'prof.']
    name_lower = name.lower()
    for title in titles:
        name_lower = re.sub(rf'\b{title}\b\.?\s*', '', name_lower)
    words = [word for word in name_lower.split() if word]
    return ' '.join(word.title() for word in words)


def are_names_matching(name1, name2):
    name1 = normalize_name(name1)
    name2 = normalize_name(name2)
    if not name1 or not name2:
        return True
    name1_parts = set(name1.split())
    name2_parts = set(name2.split())
    if name1_parts.issubset(name2_parts) or name2_parts.issubset(name1_parts):
        return True
    matching_parts = name1_parts.intersection(name2_parts)
    total_unique_parts = name1_parts.union(name2_parts)
    if len(matching_parts) >= 2 and len(matching_parts) / len(total_unique_parts) >= 0.6:
        return True
    return False


def get_most_common_or_latest(items, is_date=False):
    if not items:
        return "N/A"
    if is_date:
        parsed_dates = [parse_date_dd_mm_yyyy(d) for d in items]
        valid_parsed_dates = [d for d in parsed_dates if d is not None]
        return format_date_dd_mm_yyyy(max(valid_parsed_dates)) if valid_parsed_dates else "N/A"
    return Counter(items).most_common(1)[0][0]


def consolidate_patient_info(patient_info_list):
    if not patient_info_list:
        return {}

    names = [pi.get('name') for pi in patient_info_list if pi.get('name') and pi.get('name') not in ['N/A', '']]
    if len(names) >= 2:
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                if not are_names_matching(names[i], names[j]):
                    return {
                        'error': 'name_mismatch',
                        'names': names,
                        'conflicting_names': [names[i], names[j]]
                    }

    final_name = "N/A"
    if names:
        normalized_names = [normalize_name(name) for name in names if name]
        if normalized_names:
            name_counts = Counter(normalized_names)
            max_count = max(name_counts.values())
            most_frequent_names = [name for name, count in name_counts.items() if count == max_count]
            final_name = max(most_frequent_names, key=len) if most_frequent_names else normalized_names[0]

    final_age = "N/A"
    ages = [pi.get('age') for pi in patient_info_list if pi.get('age') and pi.get('age') not in ['N/A', '']]
    if ages:
        date_age_pairs = []
        for pi in patient_info_list:
            pi_date_str = pi.get('date')
            pi_age = pi.get('age')
            if pi_date_str and pi_date_str not in ['N/A', ''] and pi_age and pi_age not in ['N/A', '']:
                parsed_date = parse_date_dd_mm_yyyy(pi_date_str)
                if parsed_date is not None:
                    date_age_pairs.append((parsed_date, pi_age))
        if date_age_pairs:
            date_age_pairs.sort(key=lambda x: x[0], reverse=True)
            final_age = date_age_pairs[0][1]
        else:
            final_age = Counter(ages).most_common(1)[0][0]

    return {
        'name': final_name,
        'age': final_age,
        'gender': get_most_common_or_latest([pi.get('gender') for pi in patient_info_list if pi.get('gender') and pi.get('gender') not in ['N/A', '']]),
        'patient_id': get_most_common_or_latest([pi.get('patient_id') for pi in patient_info_list if pi.get('patient_id') and pi.get('patient_id') not in ['N/A', '']]),
        'date': get_most_common_or_latest([pi.get('date') for pi in patient_info_list if pi.get('date') and pi.get('date') not in ['N/A', '']], is_date=True),
        'lab_name': get_most_common_or_latest([pi.get('lab_name') for pi in patient_info_list if pi.get('lab_name') and pi.get('lab_name') not in ['N/A', '']])
    }


def smart_consolidate_patient_info(existing_patient_info, new_patient_info_list):
    """
    Intelligently consolidate patient info from existing Excel and new PDFs
    Priority: New PDF data > Existing Excel data (as PDFs are more detailed and recent)
    """

    if not new_patient_info_list and not existing_patient_info:
        return {}

    if not new_patient_info_list:
        return existing_patient_info

    if not existing_patient_info:
        # Use your existing consolidate_patient_info function that's already in your file
        return consolidate_patient_info(new_patient_info_list)

    # Both exist - merge intelligently
    consolidated = {}

    # Name: prefer from new PDFs if available, otherwise use existing
    new_names = [pi.get('name', '') for pi in new_patient_info_list if pi.get('name') and pi.get('name') != 'N/A']
    if new_names:
        # Use your existing smart name selection logic
        normalized_names = [normalize_name(name) for name in new_names if name]
        if normalized_names:
            name_counts = Counter(normalized_names)
            max_count = max(name_counts.values())
            most_frequent_names = [name for name, count in name_counts.items() if count == max_count]
            consolidated['name'] = max(most_frequent_names, key=len) if most_frequent_names else normalized_names[0]
        else:
            consolidated['name'] = existing_patient_info.get('name', 'N/A')
    else:
        consolidated['name'] = existing_patient_info.get('name', 'N/A')

    # Age: prefer from most recent PDF, otherwise use existing
    new_ages = [(pi.get('date', ''), pi.get('age', '')) for pi in new_patient_info_list if pi.get('age') and pi.get('age') != 'N/A']
    if new_ages:
        # Sort by date and get most recent age
        date_age_pairs = []
        for date_str, age in new_ages:
            if date_str:
                parsed_date = parse_date_dd_mm_yyyy(date_str)
                if parsed_date:
                    date_age_pairs.append((parsed_date, age))

        if date_age_pairs:
            date_age_pairs.sort(key=lambda x: x[0], reverse=True)
            consolidated['age'] = date_age_pairs[0][1]
        else:
            consolidated['age'] = new_ages[0][1]  # Use first available
    else:
        consolidated['age'] = existing_patient_info.get('age', 'N/A')

    # Gender: prefer from new PDFs
    new_genders = [pi.get('gender', '') for pi in new_patient_info_list if pi.get('gender') and pi.get('gender') != 'N/A']
    consolidated['gender'] = Counter(new_genders).most_common(1)[0][0] if new_genders else existing_patient_info.get('gender', 'N/A')

    # Patient ID: prefer from new PDFs
    new_ids = [pi.get('patient_id', '') for pi in new_patient_info_list if pi.get('patient_id') and pi.get('patient_id') != 'N/A']
    consolidated['patient_id'] = Counter(new_ids).most_common(1)[0][0] if new_ids else existing_patient_info.get('patient_id', 'N/A')

    # Lab name: combine both (could be different labs over time)
    new_labs = [pi.get('lab_name', '') for pi in new_patient_info_list if pi.get('lab_name') and pi.get('lab_name') != 'N/A']
    all_labs = new_labs + ([existing_patient_info.get('lab_name', '')] if existing_patient_info.get('lab_name') and existing_patient_info.get('lab_name') != 'N/A' else [])
    consolidated['lab_name'] = Counter(all_labs).most_common(1)[0][0] if all_labs else 'N/A'

    # Date: use most recent from all sources
    all_dates = []
    for pi in new_patient_info_list:
        if pi.get('date') and pi.get('date') != 'N/A':
            all_dates.append(pi.get('date'))
    if existing_patient_info.get('date') and existing_patient_info.get('date') != 'N/A':
        all_dates.append(existing_patient_info.get('date'))

    if all_dates:
        parsed_dates = [parse_date_dd_mm_yyyy(d) for d in all_dates]
        valid_dates = [d for d in parsed_dates if d is not None]
        consolidated['date'] = format_date_dd_mm_yyyy(max(valid_dates)) if valid_dates else 'N/A'
    else:
        consolidated['date'] = 'N/A'

    return consolidated


def process_existing_excel_csv(uploaded_excel_file, new_patient_info_list):
    """
    FIXED: Handle the exact Excel structure from your screenshot without creating unwanted columns
    """
    try:
        filename = uploaded_excel_file.name

        # Read the file
        if filename.endswith('.csv'):
            df = pd.read_csv(uploaded_excel_file)
            logger.debug("✓ Read CSV: %s", df.shape)
        else:
            # For Excel - read the exact structure shown in your image
            logger.debug("Reading Excel with your specific structure...")

            # Read the entire first sheet without skipping rows initially
            full_df = pd.read_excel(uploaded_excel_file, sheet_name=0, header=None)
            logger.debug("Full Excel shape: %s", full_df.shape)

            # Extract the date row (row 2, index 1)
            date_row = full_df.iloc[1, :].values

            # Extract the lab row (row 3, index 2)
            lab_row = full_df.iloc[2, :].values

            # Create column headers by combining date + lab
            new_columns = []
            for i, (date_val, lab_val) in enumerate(zip(date_row, lab_row, strict=False)):
                if i == 0:  # First column is Test_Category
                    new_columns.append('Test_Category')
                elif i == 1:  # Second column is Test_Name
                    new_columns.append('Test_Name')
                else:  # Data columns - combine date_lab
                    date_str = str(date_val).strip() if not pd.isna(date_val) else ""
                    lab_str = str(lab_val).strip() if not pd.isna(lab_val) else ""

                    # FIXED: Skip columns that are not actual data columns
                    # Skip if both date and lab are empty or contain unwanted values
                    if (not date_str or date_str.lower() in ['nan', 'col', '', 'reference', 'ranges']) and \
                       (not lab_str or lab_str.lower() in ['nan', '7', '', 'reference', 'ranges']):
                        continue

                    # Skip if this looks like a reference range column header
                    if 'reference' in date_str.lower() or 'reference' in lab_str.lower():
                        new_columns.append('Reference_Ranges')  # Keep as reference column, don't treat as data
                        continue

                    # Create column name as date_lab for actual data columns
                    if date_str and date_str != 'nan' and lab_str and lab_str != 'nan':
                        col_name = f"{date_str}_{lab_str}"
                    elif date_str and date_str != 'nan':
                        col_name = date_str
                    else:
                        continue  # Skip this column entirely

                    new_columns.append(col_name)

            logger.debug("Created column names: %s", new_columns)

            # Extract the actual data (starting from row 5, index 4)
            data_df = full_df.iloc[4:].copy()

            # FIXED: Only keep the columns we actually want
            data_df = data_df.iloc[:, :len(new_columns)]  # Trim to match our column count

            # Set the new column names
            data_df.columns = new_columns

            # Remove completely empty rows
            data_df = data_df.dropna(how='all').reset_index(drop=True)

            df = data_df
            logger.debug("✓ Processed Excel structure: %s", df.shape)
            logger.debug("Final columns: %s", list(df.columns))

        if df is None or df.empty:
            logger.debug("❌ Could not extract any data")
            return pd.DataFrame(), {}

        # Show what we extracted
        logger.debug("Final dataframe shape: %s", df.shape)

        # Process as pivoted format (since that's what your Excel structure is)
        return process_excel_pivoted_format_fixed(df, filename, new_patient_info_list)

    except Exception as e:
        logger.debug("❌ Error processing file: %s", str(e))
        import traceback
        logger.debug(traceback.format_exc())
        return pd.DataFrame(), {}


def process_excel_pivoted_format_fixed(df, filename, new_patient_info_list):
    """
    FIXED: Process the pivoted Excel format without including unwanted columns
    """
    logger.debug("Processing Excel pivoted format: %s", df.shape)

    # Get data columns - exclude Test_Category, Test_Name, and Reference_Ranges
    data_cols = []
    excluded_cols = []

    for col in df.columns:
        col_str = str(col).lower()

        # Skip the main structural columns
        if col in ['Test_Category', 'Test_Name']:
            continue

        # Skip reference range columns
        if ('reference' in col_str and 'range' in col_str) or col_str in ['reference_ranges', 'ref_range']:
            excluded_cols.append(col)
            continue

        # Skip columns that look like unwanted metadata
        if col_str in ['col', 'column', 'unnamed'] or col_str.startswith('unnamed'):
            excluded_cols.append(col)
            continue

        # This should be a data column
        data_cols.append(col)

    logger.debug("Data columns to process: %s", data_cols)
    logger.debug("Excluded columns: %s", excluded_cols)

    normalized_rows = []

    for _idx, row in df.iterrows():
        test_category = str(row.get('Test_Category', 'General')).strip()
        test_name = str(row.get('Test_Name', '')).strip()

        # Skip if no valid test name
        if not test_name or test_name.lower() in ['nan', 'test_name', ''] or pd.isna(test_name):
            continue

        # Get reference range if available
        ref_range = 'N/A'
        if 'Reference_Ranges' in df.columns:
            ref_range = str(row.get('Reference_Ranges', 'N/A')).strip()
            if ref_range.lower() in ['nan', ''] or pd.isna(ref_range):
                ref_range = 'N/A'

        # Process each data column
        for col in data_cols:
            value = row[col]

            # Skip empty values
            if pd.isna(value) or str(value).strip() in ['', 'None', 'nan']:
                continue

            # Parse date and lab from column name
            col_str = str(col)
            if '_' in col_str:
                parts = col_str.split('_', 1)
                date_part = parts[0]
                lab_part = parts[1] if len(parts) > 1 else 'Lab'
            else:
                date_part = col_str
                lab_part = 'Lab'

            # Use patient info from PDFs if available
            if new_patient_info_list:
                latest_info = new_patient_info_list[-1]
                patient_name = latest_info.get('name', 'Patient')
                patient_age = latest_info.get('age', 'N/A')
                patient_gender = latest_info.get('gender', 'N/A')
                patient_id = latest_info.get('patient_id', 'N/A')
            else:
                patient_name = extract_patient_info_from_excel_filename(filename)
                patient_age = 'N/A'
                patient_gender = 'N/A'
                patient_id = 'N/A'

            # Create normalized row
            normalized_row = {
                'Source_Filename': filename,
                'Patient_ID': patient_id,
                'Patient_Name': patient_name,
                'Age': patient_age,
                'Gender': patient_gender,
                'Test_Date': date_part,
                'Lab_Name': lab_part,
                'Test_Category': test_category,
                'Original_Test_Name': test_name,
                'Test_Name': test_name,
                'Result': str(value),
                'Unit': 'N/A',
                'Reference_Range': ref_range,  # Use the reference range from the Excel file
                'Status': 'N/A',
                'Processed_Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'Result_Numeric': parse_result_numeric(value)[0],
                'Test_Date_dt': parse_date_dd_mm_yyyy(date_part)
            }

            normalized_rows.append(normalized_row)

    logger.debug("✅ Created %s normalized rows from Excel", len(normalized_rows))

    if normalized_rows:
        result_df = pd.DataFrame(normalized_rows)

        # Create patient info
        if new_patient_info_list:
            patient_info = new_patient_info_list[-1].copy()
        else:
            patient_info = {
                'name': patient_name if 'patient_name' in locals() else 'Patient',
                'age': patient_age if 'patient_age' in locals() else 'N/A',
                'gender': patient_gender if 'patient_gender' in locals() else 'N/A',
                'patient_id': patient_id if 'patient_id' in locals() else 'N/A',
                'date': date_part if 'date_part' in locals() else 'N/A',
                'lab_name': lab_part if 'lab_part' in locals() else 'N/A'
            }

        logger.debug("✅ Successfully processed Excel data:")
        logger.debug("   - %s total records", len(result_df))
        if not result_df.empty:
            logger.debug("   - Date range: {result_df['Test_Date'].min()} to %s", result_df['Test_Date'].max())

        return result_df, patient_info
    else:
        logger.debug("❌ No data rows created from Excel")
        return pd.DataFrame(), {}
