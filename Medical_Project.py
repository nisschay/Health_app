import streamlit as st
import pandas as pd
from datetime import datetime
import re
import PyPDF2
import io
from collections import Counter
import json
import plotly.graph_objs as go
import google.generativeai as genai
from test_category_mapping import TEST_CATEGORY_TO_BODY_PARTS, BODY_PARTS_TO_EMOJI
from unify_test_names import unify_test_names
import sys
import os

# Add the script's directory to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

# --- Configuration for Standardization ---
TEST_NAME_MAPPING = {
    re.compile(r'haemoglobin', re.IGNORECASE): 'Haemoglobin',
    re.compile(r'hgb', re.IGNORECASE): 'Haemoglobin',
    re.compile(r'total leucocyte count', re.IGNORECASE): 'Total Leucocyte Count',
    re.compile(r'tlc', re.IGNORECASE): 'Total Leucocyte Count',
    re.compile(r'white blood cell count', re.IGNORECASE): 'Total Leucocyte Count',
    re.compile(r'wbc', re.IGNORECASE): 'Total Leucocyte Count',
    re.compile(r'platelet count', re.IGNORECASE): 'Platelet Count',
    re.compile(r'rbc count', re.IGNORECASE): 'Red Blood Cell Count',
    re.compile(r'red cell count', re.IGNORECASE): 'Red Blood Cell Count',
    re.compile(r'hematocrit', re.IGNORECASE): 'Hematocrit',
    re.compile(r'hct', re.IGNORECASE): 'Hematocrit',
    re.compile(r'packed cell volume', re.IGNORECASE): 'Hematocrit',
    re.compile(r'pcv', re.IGNORECASE): 'Hematocrit',
    re.compile(r'mcv', re.IGNORECASE): 'Mean Corpuscular Volume (MCV)',
    re.compile(r'mch', re.IGNORECASE): 'Mean Corpuscular Haemoglobin (MCH)',
    re.compile(r'mchc', re.IGNORECASE): 'Mean Corpuscular Haemoglobin Concentration (MCHC)',
    re.compile(r'rdw-cv', re.IGNORECASE): 'Red Cell Distribution Width - CV (RDW-CV)',
    re.compile(r'rdw-sd', re.IGNORECASE): 'Red Cell Distribution Width - SD (RDW-SD)',
    re.compile(r'neutrophils?', re.IGNORECASE): 'Neutrophils',
    re.compile(r'lymphocytes?', re.IGNORECASE): 'Lymphocytes',
    re.compile(r'monocytes?', re.IGNORECASE): 'Monocytes',
    re.compile(r'eosinophils?', re.IGNORECASE): 'Eosinophils',
    re.compile(r'basophils?', re.IGNORECASE): 'Basophils',
    re.compile(r'esr', re.IGNORECASE): 'Erythrocyte Sedimentation Rate (ESR)',
    re.compile(r'total bilirubin', re.IGNORECASE): 'Bilirubin - Total',
    re.compile(r'direct bilirubin', re.IGNORECASE): 'Bilirubin - Direct',
    re.compile(r'indirect bilirubin', re.IGNORECASE): 'Bilirubin - Indirect',
    re.compile(r'sgpt', re.IGNORECASE): 'Alanine Aminotransferase (SGPT/ALT)',
    re.compile(r'alt', re.IGNORECASE): 'Alanine Aminotransferase (SGPT/ALT)',
    re.compile(r'sgot', re.IGNORECASE): 'Aspartate Aminotransferase (SGOT/AST)',
    re.compile(r'ast', re.IGNORECASE): 'Aspartate Aminotransferase (SGOT/AST)',
    re.compile(r'alkaline phosphatase', re.IGNORECASE): 'Alkaline Phosphatase (ALP)',
    re.compile(r'alp', re.IGNORECASE): 'Alkaline Phosphatase (ALP)',
    re.compile(r'total protein', re.IGNORECASE): 'Protein - Total',
    re.compile(r'albumin', re.IGNORECASE): 'Albumin',
    re.compile(r'globulin', re.IGNORECASE): 'Globulin',
    re.compile(r'a/g ratio', re.IGNORECASE): 'Albumin/Globulin Ratio (A/G Ratio)',
    re.compile(r'urea', re.IGNORECASE): 'Urea',
    re.compile(r'blood urea nitrogen', re.IGNORECASE): 'Blood Urea Nitrogen (BUN)',
    re.compile(r'bun', re.IGNORECASE): 'Blood Urea Nitrogen (BUN)',
    re.compile(r'creatinine', re.IGNORECASE): 'Creatinine',
    re.compile(r'uric acid', re.IGNORECASE): 'Uric Acid',
    re.compile(r'total cholesterol', re.IGNORECASE): 'Cholesterol - Total',
    re.compile(r'triglycerides', re.IGNORECASE): 'Triglycerides',
    re.compile(r'hdl cholesterol', re.IGNORECASE): 'HDL Cholesterol',
    re.compile(r'ldl cholesterol', re.IGNORECASE): 'LDL Cholesterol',
    re.compile(r'vldl cholesterol', re.IGNORECASE): 'VLDL Cholesterol',
    re.compile(r'cholesterol/hdl ratio', re.IGNORECASE): 'Total Cholesterol/HDL Ratio',
    re.compile(r'glucose fasting', re.IGNORECASE): 'Glucose - Fasting',
    re.compile(r'fbs', re.IGNORECASE): 'Glucose - Fasting',
    re.compile(r'glucose random', re.IGNORECASE): 'Glucose - Random',
    re.compile(r'rbs', re.IGNORECASE): 'Glucose - Random',
    re.compile(r'hba1c', re.IGNORECASE): 'Glycated Haemoglobin (HbA1c)',
    re.compile(r'tsh', re.IGNORECASE): 'Thyroid Stimulating Hormone (TSH)',
    re.compile(r'total t3', re.IGNORECASE): 'Total T3',
    re.compile(r'total t4', re.IGNORECASE): 'Total T4',
    re.compile(r'free t3', re.IGNORECASE): 'Free T3 (FT3)',
    re.compile(r'ft3', re.IGNORECASE): 'Free T3 (FT3)',
    re.compile(r'free t4', re.IGNORECASE): 'Free T4 (FT4)',
    re.compile(r'ft4', re.IGNORECASE): 'Free T4 (FT4)',
}

UNIT_MAPPING = {
    re.compile(r'gm/dl', re.IGNORECASE): 'g/dL',
    re.compile(r'g/l', re.IGNORECASE): 'g/L',
    re.compile(r'cells/cumm', re.IGNORECASE): 'cells/µL',
    re.compile(r'/cumm', re.IGNORECASE): '/µL',
    re.compile(r'thou/cumm', re.IGNORECASE): 'x10³/µL',
    re.compile(r'10\^3/ul', re.IGNORECASE): 'x10³/µL',
    re.compile(r'x10\^3/μl', re.IGNORECASE): 'x10³/µL',
    re.compile(r'mill/cumm', re.IGNORECASE): 'x10⁶/µL',
    re.compile(r'10\^6/ul', re.IGNORECASE): 'x10⁶/µL',
    re.compile(r'x10\^6/μl', re.IGNORECASE): 'x10⁶/µL',
    re.compile(r'mg/dl', re.IGNORECASE): 'mg/dL',
    re.compile(r'iu/l', re.IGNORECASE): 'IU/L',
    re.compile(r'u/l', re.IGNORECASE): 'U/L',
    re.compile(r'µiu/ml', re.IGNORECASE): 'µIU/mL',
    re.compile(r'ng/dl', re.IGNORECASE): 'ng/dL',
    re.compile(r'pg/ml', re.IGNORECASE): 'pg/mL',
    re.compile(r'%', re.IGNORECASE): '%',
    re.compile(r'fl', re.IGNORECASE): 'fL',
    re.compile(r'pg', re.IGNORECASE): 'pg',
    re.compile(r'mm/hr', re.IGNORECASE): 'mm/hr',
}

STATUS_MAPPING = {
    re.compile(r'low', re.IGNORECASE): 'Low',
    re.compile(r'normal', re.IGNORECASE): 'Normal',
    re.compile(r'high', re.IGNORECASE): 'High',
    re.compile(r'critical', re.IGNORECASE): 'Critical',
    re.compile(r'n/a', re.IGNORECASE): 'N/A',
    re.compile(r'not applicable', re.IGNORECASE): 'N/A',
    re.compile(r'within normal limits', re.IGNORECASE): 'Normal',
    re.compile(r'within reference range', re.IGNORECASE): 'Normal',
    re.compile(r'near optimal', re.IGNORECASE): 'Near Optimal',
    re.compile(r'desirable', re.IGNORECASE): 'Desirable',
    re.compile(r'borderline high', re.IGNORECASE): 'Borderline High',
}

# Initialize Gemini models globally
gemini_model_extraction = None
gemini_model_chat = None

# --- Helper Functions ---
def parse_date_dd_mm_yyyy(date_str):
    """Parse date string ensuring DD/MM/YYYY format"""
    if not date_str or date_str in ['N/A', '']:
        return None
    
    try:
        # First try to parse as DD/MM/YYYY
        return pd.to_datetime(date_str, format='%d/%m/%Y', errors='raise')
    except:
        try:
            # Try DD-MM-YYYY
            return pd.to_datetime(date_str, format='%d-%m-%Y', errors='raise')
        except:
            try:
                # Try DD.MM.YYYY
                return pd.to_datetime(date_str, format='%d.%m.%Y', errors='raise')
            except:
                try:
                    # If all else fails, try to parse and assume it's in DD/MM/YYYY format
                    # Force dayfirst=True to ensure DD/MM/YYYY interpretation
                    return pd.to_datetime(date_str, dayfirst=True, errors='raise')
                except:
                    return None

def format_date_dd_mm_yyyy(date_obj):
    """Format datetime object to DD-MM-YYYY string"""
    if pd.isna(date_obj) or date_obj is None:
        return 'N/A'
    return date_obj.strftime('%d-%m-%Y')

def standardize_value(value, mapping_dict, default_case='title'):
    if not isinstance(value, str):
        return value
    original_value = value.strip()
    for pattern, standard_form in mapping_dict.items():
        if pattern.search(original_value):
            if default_case == 'title':
                return standard_form.title()
            elif default_case == 'lower':
                return standard_form.lower()
            return standard_form
    if default_case == 'title':
        return original_value.title()
    elif default_case == 'lower':
        return original_value.lower()
    return original_value


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

def extract_text_from_pdf(file_content):
    try:
        pdf_file = io.BytesIO(file_content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page_num, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            if page_text:
                text += f"\n--- Page {page_num+1} ---\n{page_text}"
        if not text.strip():
            st.warning("Warning: No text extracted from PDF. PDF might be image-based or empty.")
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return None

def init_gemini_models(api_key_for_gemini):
    global gemini_model_extraction, gemini_model_chat
    try:
        if not api_key_for_gemini:
            st.error("Gemini API Key is missing. Cannot initialize models.")
            return False
        
        genai.configure(api_key=api_key_for_gemini)
        
        gemini_model_extraction = genai.GenerativeModel('gemini-2.5-flash')
        gemini_model_chat = genai.GenerativeModel('gemini-2.5-flash')
        
        st.success("Successfully connected to Gemini API")
        return True
    except Exception as e:
        st.error(f"Error configuring Gemini: {e}. Please ensure your API key is correct and valid.")
        gemini_model_extraction = None
        gemini_model_chat = None
        return False

def analyze_medical_report_with_gemini(text_content, api_key_for_gemini):
    global gemini_model_extraction
    if not gemini_model_extraction and not init_gemini_models(api_key_for_gemini):
        st.error("Gemini extraction model not initialized. API key might be missing or invalid.")
        return None

    if not text_content or not text_content.strip():
        st.warning("No text provided to analyze.")
        return None

    prompt = f"""
    Analyze this medical test report. Extract all patient information and test results.
    The patient's full name is critical. Prioritize complete and formal names (e.g., "Nisschay Khandelwal" over "SelfNisschay Khandelwal" or "N Khandelwal"). Avoid prefixes like "Self" if a clearer name is available.
    Also extract Patient ID, Age (e.g., "35 years", "35 Y", "35"), and Gender (e.g., "Male", "Female", "M", "F").
    The report date or collection date is also critical. IMPORTANT: Ensure date is in DD-MM-YYYY format (day first, then month, then year). If you see a date like "15/03/2024" or "15-03-2024", this should be interpreted as 15th March 2024, not 3rd month 15th day. If multiple dates are present (collection, report), prefer collection date.
    IMPORTANT: Extract the main laboratory/hospital name that conducted the tests. Look for prominent facility names like "Neuberg", "Apollo Hospital", "Quest Diagnostics", "Dr. Lal PathLabs", etc.
    - This is usually prominently displayed at the top of the report as the main facility name
    - Ignore billing locations, collection centers, or subsidiary names in smaller text
    - Look for the main brand/facility name that appears in large text or as a header
    - If you see names like "Neuberg Abha", "Apollo Hospitals", "Max Healthcare" etc., prefer these over technical/billing names
    - Avoid names that look like billing addresses or subsidiary locations

    For each test parameter, extract:
    - Test Name (e.g., "Haemoglobin", "Total Leucocyte Count")
    - Result (numerical value or finding like "Detected", "Not Detected", "Positive", "Negative")
    - Unit of measurement (e.g., "g/dL", "cells/µL")
    - Reference Range (e.g., "13.0 - 17.0", "< 5.0", "Negative")
    - Status (interpret as "Low", "Normal", "High", "Critical", "Positive", "Negative", or "N/A" if not applicable or clearly stated. If not stated, use "N/A")
    - Category (e.g., "Haematology", "Liver Function Test", "Kidney Function Test", "Lipid Profile", "Thyroid Profile", "Urinalysis". Infer if not explicitly stated.)

    Return the data in this exact JSON format:
    {{
        "patient_info": {{
            "name": "Full Patient Name",
            "age": "Age",
            "gender": "Gender",
            "patient_id": "Patient ID or Registration No.",
            "date": "Test Date or Report Date (DD-MM-YYYY)",
            "lab_name": "Laboratory or Medical Center Name"
        }},
        "test_results": [
            {{
                "test_name": "Name of the test",
                "result": "Numerical value or finding",
                "unit": "Unit of measurement",
                "reference_range": "Normal reference range",
                "status": "Low/Normal/High/Critical/Positive/Negative/N/A",
                "category": "Test category"
            }}
        ],
        "abnormal_findings_summary_from_report": [
            "List any general abnormal findings or summary remarks directly stated in the report, if present."
        ]
    }}

    Medical Report Text:
    ---
    {text_content}
    ---
    """
    try:
        response = gemini_model_extraction.generate_content(prompt)
        response_text = response.text
        
        match_json_block = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if match_json_block:
            json_str = match_json_block.group(1)
        else:
            start_index = response_text.find('{')
            end_index = response_text.rfind('}')
            if start_index != -1 and end_index > start_index:
                json_str = response_text[start_index : end_index+1]
            else:
                st.error("Could not find a clear JSON block in Gemini API response.")
                st.text_area("Gemini API Response (text):", response_text, height=150)
                return None
        
        return json.loads(json_str)
    except json.JSONDecodeError as json_e:
        st.error(f"Error decoding JSON from Gemini response: {json_e}")
        st.text_area("Problematic JSON string:", json_str, height=150)
        st.text_area("Full Gemini Response (text):", response_text, height=150)
        return None
    except Exception as e:
        st.error(f"Error analyzing report with Gemini: {str(e)}")
        if "API key not valid" in str(e) or "PERMISSION_DENIED" in str(e):
            st.error("Please ensure your Gemini API key is correct and has the necessary permissions.")
        return None

def create_structured_dataframe(ai_results_json, source_filename="Uploaded PDF"):
    if not ai_results_json or 'test_results' not in ai_results_json:
        return pd.DataFrame(), {}

    patient_info_dict = ai_results_json.get('patient_info', {})
    report_date_str = patient_info_dict.get('date', 'N/A')
    parsed_date = 'N/A'
    if report_date_str and report_date_str != 'N/A':
        dt = parse_date_dd_mm_yyyy(report_date_str)
        if dt is not None:
            parsed_date = format_date_dd_mm_yyyy(dt)
    patient_info_dict['date'] = parsed_date

    all_rows = []
    for test_result in ai_results_json.get('test_results', []):
        row = {
            'Source_Filename': source_filename,
            'Patient_ID': patient_info_dict.get('patient_id', 'N/A'),
            'Patient_Name': patient_info_dict.get('name', 'N/A'),
            'Age': patient_info_dict.get('age', 'N/A'),
            'Gender': patient_info_dict.get('gender', 'N/A'),
            'Test_Date': parsed_date,
            'Lab_Name': patient_info_dict.get('lab_name', 'N/A'),
            'Test_Category': standardize_value(test_result.get('category', 'N/A'), {}, default_case='title'),
            'Original_Test_Name': test_result.get('test_name', 'UnknownTest'),
            'Test_Name': standardize_value(test_result.get('test_name', 'UnknownTest'), TEST_NAME_MAPPING, default_case='title'),
            'Result': test_result.get('result', ''),
            'Unit': standardize_value(test_result.get('unit', ''), UNIT_MAPPING, default_case='original'),
            'Reference_Range': test_result.get('reference_range', ''),
            'Status': standardize_value(test_result.get('status', ''), STATUS_MAPPING, default_case='title'),
            'Processed_Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        all_rows.append(row)

    if not all_rows:
        return pd.DataFrame(), patient_info_dict

    df = pd.DataFrame(all_rows)
    df['Result_Numeric'] = pd.to_numeric(df['Result'], errors='coerce')
    # Use the new date parsing function
    df['Test_Date_dt'] = df['Test_Date'].apply(parse_date_dd_mm_yyyy)
    df = df.sort_values(by=['Test_Date_dt', 'Test_Category', 'Test_Name']).reset_index(drop=True)
    
    return df, patient_info_dict

def normalize_name(name):
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

def get_chatbot_response(report_df_for_prompt, user_question, chat_history_for_prompt, api_key_for_gemini):
    global gemini_model_chat
    if not gemini_model_chat and not init_gemini_models(api_key_for_gemini):
        return "Chatbot model not initialized. API key might be missing or invalid."
    if report_df_for_prompt.empty:
        return "No report data available to answer questions. Please analyze a report first."
    
    df_string = report_df_for_prompt[['Test_Date', 'Test_Category', 'Test_Name', 'Result', 'Unit', 'Reference_Range', 'Status']].to_string(index=False, max_rows=50)
    
    history_context = ""
    for entry in chat_history_for_prompt[-5:]:
        history_context += f"{entry['role'].capitalize()}: {entry['content']}\n"

    prompt = f"""You are a medical report assistant, so try to be precise.
Based on the provided medical test results and chat history, answer the user's question.
Important guidelines:
- Structure your response in clear, concise bullet points
- Present one piece of information per bullet point
- Use sub-bullets for additional details when needed
- Keep explanations brief and focused
- Do not provide medical advice or diagnosis
- Stick to summarizing and explaining the data
- For abnormal values, include the reference range in brackets
- If answering about overall health, categorize findings as: Normal, Borderline, or Concerning
- If the question is about a specific test, focus on that test's results and context
- Explain medical jargon in simple terms please
- If the question is about trends, summarize changes over time for relevant tests

Chat History:
{history_context}

Available Medical Report Data (summary):
{df_string}

User Question: {user_question}

Assistant Response (please use bullet points):
"""
    try:
        response = gemini_model_chat.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Error getting chatbot response from Gemini: {str(e)}")
        return "Sorry, I encountered an error trying to respond."

def parse_reference_range(ref_range_str):
    if not isinstance(ref_range_str, str) or ref_range_str.lower() == 'n/a' or not ref_range_str.strip():
        return None, None, None
    
    ref_range_str = ref_range_str.strip()
    
    match_range = re.search(r'([\d.]+)\s*-\s*([\d.]+)', ref_range_str)
    if match_range:
        try: return float(match_range.group(1)), float(match_range.group(2)), "range"
        except ValueError: pass

    match_less_than = re.search(r'(?:<|Less than|upto)\s*([\d.]+)', ref_range_str, re.IGNORECASE)
    if match_less_than:
        try: return None, float(match_less_than.group(1)), "less_than"
        except ValueError: pass

    match_greater_than = re.search(r'(?:>|Greater than|above)\s*([\d.]+)', ref_range_str, re.IGNORECASE)
    if match_greater_than:
        try: return float(match_greater_than.group(1)), None, "greater_than"
        except ValueError: pass
    
    if ref_range_str.lower() in ["negative", "non reactive", "not detected"]:
        return None, None, "qualitative_normal"
    if ref_range_str.lower() in ["positive", "reactive", "detected"]:
        return None, None, "qualitative_abnormal"
        
    return None, None, None

def generate_test_plot(df_report, selected_test_name, selected_date=None):
    test_data_for_plot = df_report[df_report['Test_Name'] == selected_test_name]
    
    if selected_date and selected_date != "All Dates":
        test_data_for_plot = test_data_for_plot[test_data_for_plot['Test_Date'] == selected_date]
    
    if test_data_for_plot.empty:
        st.warning(f"No data found for test: {selected_test_name}" + (f" on {selected_date}" if selected_date and selected_date != "All Dates" else ""))
        return None

    if "All Dates" == selected_date and len(test_data_for_plot['Test_Date_dt'].unique()) > 1:
        fig = go.Figure()
        test_data_for_plot = test_data_for_plot.sort_values('Test_Date_dt')
        
        if pd.to_numeric(test_data_for_plot['Result'], errors='coerce').notna().all():
            # Format dates for display using DD-MM-YYYY
            test_data_for_plot['Date_Display'] = test_data_for_plot['Test_Date_dt'].apply(format_date_dd_mm_yyyy)
            
            fig.add_trace(go.Scatter(
                x=test_data_for_plot['Date_Display'],
                y=test_data_for_plot['Result_Numeric'],
                mode='lines+markers',
                name='Result Trend',
                line=dict(width=3),
                marker=dict(size=8)
            ))
            
            latest_entry = test_data_for_plot.iloc[-1]
            low_ref, high_ref, ref_type = parse_reference_range(latest_entry['Reference_Range'])
            
            data_min = test_data_for_plot['Result_Numeric'].min()
            data_max = test_data_for_plot['Result_Numeric'].max()
            data_range = data_max - data_min
            
            if ref_type == "range" and low_ref is not None and high_ref is not None:
                fig.add_hline(y=high_ref, line_dash="dash", line_color="red", annotation_text="Upper Reference", annotation_position="bottom right")
                fig.add_hline(y=low_ref, line_dash="dash", line_color="green", annotation_text="Lower Reference", annotation_position="top right")
                fig.add_hrect(y0=low_ref, y1=high_ref, fillcolor="green", opacity=0.1, annotation_text="Normal Range", annotation_position="top left")
                y_min = min(data_min, low_ref) - abs(data_range * 0.1 if data_range > 0 else low_ref * 0.1)
                y_max = max(data_max, high_ref) + abs(data_range * 0.1 if data_range > 0 else high_ref * 0.1)
            elif ref_type == "less_than" and high_ref is not None:
                fig.add_hline(y=high_ref, line_dash="dash", line_color="red", annotation_text=f"< {high_ref}", annotation_position="bottom right")
                y_min = min(data_min, 0) - abs(data_range * 0.1 if data_range > 0 else data_min * 0.1)
                y_max = max(data_max, high_ref) + abs(data_range * 0.1 if data_range > 0 else high_ref * 0.1)
            elif ref_type == "greater_than" and low_ref is not None:
                fig.add_hline(y=low_ref, line_dash="dash", line_color="green", annotation_text=f"> {low_ref}", annotation_position="top right")
                y_min = min(data_min, low_ref) - abs(data_range * 0.1 if data_range > 0 else low_ref * 0.1)
                y_max = max(data_max, data_max * 1.2) + abs(data_range * 0.1 if data_range > 0 else data_max * 0.1)
            else:
                padding = abs(data_range * 0.15 if data_range > 0 else data_max * 0.15)
                y_min = data_min - padding
                y_max = data_max + padding
            
            unit = latest_entry['Unit']
            fig.update_layout(
                title_text=f"{selected_test_name} Trend ({unit})",
                xaxis_title="Date",
                yaxis_title=f"Result ({unit})",
                yaxis=dict(range=[y_min, y_max]),
                height=500,
                margin=dict(l=20, r=20, t=60, b=80),
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                autosize=True,
                xaxis=dict(tickangle=45, tickfont=dict(size=10))
            )
            return fig
        else:
            st.info(f"Cannot plot trend for '{selected_test_name}' as some results are non-numeric.")
            return None

    test_entry = test_data_for_plot.sort_values('Test_Date_dt', ascending=False).iloc[0]
    result_val_numeric = test_entry['Result_Numeric']
    result_val_str = test_entry['Result']
    ref_range_str = test_entry['Reference_Range']
    unit = test_entry['Unit']
    status = test_entry['Status']

    if pd.isna(result_val_numeric):
        st.info(f"Result for '{selected_test_name}' is '{result_val_str}' (non-numeric). Status: {status}")
        return None

    fig = go.Figure()
    
    low_ref, high_ref, ref_type = parse_reference_range(ref_range_str)
    
    if ref_type == "range" and low_ref is not None and high_ref is not None:
        ref_range_span = high_ref - low_ref
        axis_min = min(low_ref - ref_range_span * 0.2, result_val_numeric - ref_range_span * 0.2)
        axis_max = max(high_ref + ref_range_span * 0.2, result_val_numeric + ref_range_span * 0.2)
        steps = [
            {'range': [axis_min, low_ref], 'color': 'lightcoral'},
            {'range': [low_ref, high_ref], 'color': 'lightgreen'},
            {'range': [high_ref, axis_max], 'color': 'lightcoral'}
        ]
    elif ref_type == "less_than" and high_ref is not None:
        axis_min = 0
        axis_max = max(high_ref * 1.5, result_val_numeric * 1.2)
        steps = [
            {'range': [axis_min, high_ref], 'color': 'lightgreen'},
            {'range': [high_ref, axis_max], 'color': 'lightcoral'}
        ]
    elif ref_type == "greater_than" and low_ref is not None:
        axis_min = min(low_ref * 0.5, result_val_numeric * 0.8)
        axis_max = max(low_ref * 1.5, result_val_numeric * 1.2)
        steps = [
            {'range': [axis_min, low_ref], 'color': 'lightcoral'},
            {'range': [low_ref, axis_max], 'color': 'lightgreen'}
        ]
    else:
        value_range = abs(result_val_numeric * 0.5) if result_val_numeric != 0 else 10
        axis_min = result_val_numeric - value_range
        axis_max = result_val_numeric + value_range
        steps = [
            {'range': [axis_min, axis_max], 'color': 'lightblue'}
        ]

    fig.add_trace(go.Indicator(
        mode="number+gauge",
        value=result_val_numeric,
        domain={'x': [0.1, 1], 'y': [0.2, 0.9]},
        number={'suffix': f" {unit}", 'font': {'size': 20}, 'valueformat': '.2f'},
        title={
            'text': f"<br>{selected_test_name}<br><span style='font-size:0.8em;color:gray'>Reference: {ref_range_str}<br>Status: {status}</span>",
            'font': {"size": 12},
            'align': "center"
        },
        gauge={
            'shape': "bullet",
            'axis': {'range': [axis_min, axis_max]},
            'threshold': {
                'line': {'color': "red", 'width': 3},
                'thickness': 0.75,
                'value': result_val_numeric
            },
            'steps': steps,
            'bar': {'color': "rgba(0,0,0,0)"}
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=50, b=10),
        autosize=True
    )
    return fig

def create_enhanced_excel_with_trends(organized_df, ref_range_df, date_lab_cols_sorted, patient_info):
    output_excel = io.BytesIO()
    with pd.ExcelWriter(output_excel, engine='xlsxwriter') as writer:
        # Write the data without headers first, starting from row 4 (index 3)
        organized_df.to_excel(writer, index=False, sheet_name='Medical Data with Trends', startrow=3, header=False)
        
        workbook = writer.book
        worksheet = writer.sheets['Medical Data with Trends']
        
        # Title format
        title_format = workbook.add_format({'bold': True, 'font_size': 16, 'align': 'center', 'bg_color': '#4472C4', 'font_color': 'white'})
        worksheet.merge_range('A1:' + chr(65 + len(organized_df.columns) - 1) + '1', 'Medical Test Results - Organized by Date', title_format)
        
        # Header formats
        date_format = workbook.add_format({'bold': True, 'bg_color': '#E7E6E6', 'border': 1, 'align': 'center', 'font_color': '#2E5C8F'})
        lab_format = workbook.add_format({'bold': True, 'bg_color': '#F0F8FF', 'border': 1, 'align': 'center', 'font_color': '#1E7B3E', 'italic': True})
        empty_format = workbook.add_format({'border': 1, 'bg_color': '#FFFFFF'})
        
        # Write date row (row 2)
        worksheet.write(1, 0, '📅 Date', date_format)
        worksheet.write(1, 1, '', date_format)
        
        # Write lab row (row 3)
        worksheet.write(2, 0, '🏥 Lab', lab_format)
        worksheet.write(2, 1, '', lab_format)
        
        # Fill in date and lab information for data columns
        for i, date_lab_col in enumerate(date_lab_cols_sorted):
            col_idx = i + 2
            parts = date_lab_col.split('_', 1)
            date_part = parts[0] if len(parts) > 0 else 'N/A'
            lab_part = parts[1] if len(parts) > 1 else 'N/A'
            worksheet.write(1, col_idx, date_part, date_format)
            worksheet.write(2, col_idx, lab_part, lab_format)
        
        # Add reference ranges column
        ref_col_idx = len(organized_df.columns)
        worksheet.write(1, ref_col_idx, '', empty_format)  # Empty instead of date
        worksheet.write(2, ref_col_idx, '', empty_format)  # Empty instead of lab
        
        # Write custom column headers (row 4) - clean headers without the date_lab combinations
        header_format = workbook.add_format({'bold': True, 'bg_color': '#D9E2F3', 'border': 1, 'align': 'center', 'valign': 'vcenter'})
        
        # Write the first two column headers manually
        worksheet.write(3, 0, 'Test_Category', header_format)
        worksheet.write(3, 1, 'Test_Name', header_format)
        
        # Write empty headers for the data columns (instead of the date_lab combinations)
        for i, date_lab_col in enumerate(date_lab_cols_sorted):
            col_idx = i + 2
            worksheet.write(3, col_idx, '', header_format)  # Empty header instead of date_lab combination
        
        # Add "Reference Ranges" header
        worksheet.write(3, ref_col_idx, "Reference Ranges", header_format)
        
        # Data formatting
        data_format = workbook.add_format({'border': 1, 'align': 'center', 'valign': 'vcenter'})
        numeric_format = workbook.add_format({'border': 1, 'align': 'center', 'valign': 'vcenter', 'num_format': '0.00'})
        ref_format = workbook.add_format({'border': 1, 'align': 'left', 'valign': 'vcenter', 'text_wrap': True})
        
        # Write data manually with proper formatting
        for row_num in range(len(organized_df)):
            for col_num in range(len(organized_df.columns)):
                value = organized_df.iloc[row_num, col_num]
                if col_num < 2:  # Category and Test Name columns
                    worksheet.write(row_num + 4, col_num, value, data_format)
                else:  # Data columns
                    try:
                        if pd.notna(value) and str(value).strip() != '':
                            # Try to convert to float for proper numeric storage
                            float_val = float(value)
                            worksheet.write_number(row_num + 4, col_num, float_val, numeric_format)
                        else:
                            worksheet.write(row_num + 4, col_num, value if pd.notna(value) else '', data_format)
                    except (ValueError, TypeError):
                        # If conversion fails, write as text
                        worksheet.write(row_num + 4, col_num, str(value) if pd.notna(value) else '', data_format)
        
        # Write reference ranges for each row
        for row_num in range(len(organized_df)):
            # Add reference ranges from ref_range_df
            try:
                # Get all non-null reference ranges for this test across all dates
                ref_values = []
                for col_idx, col_name in enumerate(date_lab_cols_sorted):
                    if col_idx + 2 < len(ref_range_df.columns):
                        ref_value = ref_range_df.iloc[row_num, col_idx + 2]
                        if pd.notna(ref_value) and str(ref_value).strip() != '' and str(ref_value) != 'N/A':
                            ref_values.append(str(ref_value))
                
                # Use the most common reference range or the first valid one
                if ref_values:
                    from collections import Counter
                    most_common_ref = Counter(ref_values).most_common(1)[0][0]
                    worksheet.write(row_num + 4, ref_col_idx, most_common_ref, ref_format)
                else:
                    worksheet.write(row_num + 4, ref_col_idx, '', ref_format)
                    
            except Exception as e:
                worksheet.write(row_num + 4, ref_col_idx, '', ref_format)
        
        # Set row heights
        worksheet.set_row(1, 25)
        worksheet.set_row(2, 25)
        worksheet.set_row(3, 35)
        for row_num in range(len(organized_df)):
            worksheet.set_row(row_num + 4, 25)
        
        # Set column widths
        worksheet.set_column('A:A', 20)  # Test Category
        worksheet.set_column('B:B', 25)  # Test Name
        for i, col in enumerate(date_lab_cols_sorted):
            worksheet.set_column(i + 2, i + 2, 12)  # Data columns
        worksheet.set_column(ref_col_idx, ref_col_idx, 20)  # Reference ranges column
        
        # Add autofilter and freeze panes
        worksheet.autofilter(3, 0, len(organized_df) + 3, len(organized_df.columns))
        worksheet.freeze_panes(4, 2)
        
        # Create summary sheet
        summary_sheet = workbook.add_worksheet('Summary')
        summary_sheet.merge_range('A1:D1', 'Medical Report Summary', title_format)
        
        info_format = workbook.add_format({'bold': True, 'bg_color': '#E7E6E6'})
        
        # Patient information
        summary_sheet.write('A3', 'Patient Information:', info_format)
        summary_sheet.write('A4', f"Name: {patient_info.get('name', 'N/A')}")
        summary_sheet.write('A5', f"Age: {patient_info.get('age', 'N/A')}")
        summary_sheet.write('A6', f"Gender: {patient_info.get('gender', 'N/A')}")
        summary_sheet.write('A7', f"Patient ID: {patient_info.get('patient_id', 'N/A')}")
        summary_sheet.write('A8', f"Primary Lab: {patient_info.get('lab_name', 'N/A')}")
        
        # Report statistics
        summary_sheet.write('A10', 'Report Statistics:', info_format)
        summary_sheet.write('A11', f"Total Test Categories: {organized_df['Test_Category'].nunique()}")
        summary_sheet.write('A12', f"Total Tests: {len(organized_df)}")
        summary_sheet.write('A13', f"Total Date-Lab Combinations: {len(date_lab_cols_sorted)}")
        
        if date_lab_cols_sorted:
            first_date = date_lab_cols_sorted[0].split('_')[0]
            last_date = date_lab_cols_sorted[-1].split('_')[0]
            summary_sheet.write('A14', f"Date Range: {first_date} to {last_date}")
        
        summary_sheet.write('A15', f"Generated on: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}")
        
        # Test categories
        summary_sheet.write('A17', 'Test Categories:', info_format)
        categories = organized_df['Test_Category'].value_counts()
        for i, (category, count) in enumerate(categories.items()):
            summary_sheet.write(f'A{18+i}', f"• {category}: {count} tests")
        
        # Labs used
        summary_sheet.write('A' + str(18 + len(categories) + 2), 'Labs Used:', info_format)
        unique_labs = set()
        for col in date_lab_cols_sorted:
            if '_' in col:
                lab_part = col.split('_', 1)[1]
                unique_labs.add(lab_part)
        
        for i, lab in enumerate(sorted(unique_labs)):
            summary_sheet.write(f'A{18 + len(categories) + 3 + i}', f"• {lab}")

    return output_excel.getvalue()

def combine_duplicate_tests(df):
    df = df.copy()
    test_cat_counts = df.groupby(['Test_Name', 'Test_Category'])['Test_Date'].nunique().reset_index().rename(columns={'Test_Date': 'date_count'})
    test_cat_total = df.groupby(['Test_Name', 'Test_Category']).size().reset_index(name='row_count')
    merged = pd.merge(test_cat_counts, test_cat_total, on=['Test_Name', 'Test_Category'])
    
    def pick_category(subdf):
        max_dates = subdf['date_count'].max()
        date_winners = subdf[subdf['date_count'] == max_dates]
        if len(date_winners) == 1:
            return date_winners.iloc[0]['Test_Category']
        max_rows = date_winners['row_count'].max()
        row_winners = date_winners[date_winners['row_count'] == max_rows]
        return row_winners.iloc[0]['Test_Category']

    best_cats = merged.groupby('Test_Name').apply(pick_category).reset_index()
    best_cats.columns = ['Test_Name', 'Best_Test_Category']
    df = pd.merge(df, best_cats, on='Test_Name', how='left')
    df['Test_Category'] = df['Best_Test_Category']
    df = df.drop(columns=['Best_Test_Category'])
    df = df.sort_values(['Test_Name', 'Test_Date', 'Test_Category']).drop_duplicates(['Test_Name', 'Test_Date'])
    df = df.reset_index(drop=True)
    return df

# --- Streamlit App UI ---
st.set_page_config(page_title="Medical Report Analyzer", layout="wide")

# Load custom CSS
# with open('style.css') as f:
#     st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.title("⚕️ Medical Report Analyzer & Health Insights")
st.markdown("Upload your medical PDF reports to get structured data, a health summary, and visualizations.")

# --- Gemini API Key ---
api_key = st.secrets.get("GEMINI_API_KEY")
if not api_key or not init_gemini_models(api_key):
    st.error("🚨 Unable to initialize Gemini models. Please check the server configuration.")

# --- Initialize session state ---
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
if 'report_df' not in st.session_state:
    st.session_state.report_df = pd.DataFrame()
if 'consolidated_patient_info' not in st.session_state:
    st.session_state.consolidated_patient_info = {}
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'raw_texts' not in st.session_state:
    st.session_state.raw_texts = []

# --- UI for Upload ---
upload_mode = st.radio(
    "Select what you want to do:",
    ["Upload new medical reports", "Add new medical reports to an existing Excel/CSV file"],
    index=0,
    key="upload_mode_selector"
)

uploaded_excel_file_object = None
if upload_mode == "Add new medical reports to an existing Excel/CSV file":
    uploaded_excel_file_object = st.file_uploader(
        "Upload your previously downloaded Excel or CSV file",
        type=["csv", "xlsx"],
        key="excel_uploader"
    )

uploaded_files = st.file_uploader(
    "📄 Upload Medical Report PDFs (multiple allowed)", 
    type="pdf", 
    accept_multiple_files=True,
    key="pdf_uploader"
)

if st.button("🔬 Analyze Reports", key="analyze_btn"):
    st.session_state.analysis_done = False
    st.session_state.report_df = pd.DataFrame()
    st.session_state.consolidated_patient_info = {}
    st.session_state.chat_history = []
    st.session_state.raw_texts = []

    all_dfs = []
    all_patient_infos_from_pdfs = []

    with st.spinner("Processing reports and analyzing with AI... This may take a moment."):
        if uploaded_files:
            for uploaded_file in uploaded_files:
                st.write(f"--- Processing: {uploaded_file.name} ---")
                file_content = uploaded_file.read()
                report_text = extract_text_from_pdf(file_content)
                st.session_state.raw_texts.append({"name": uploaded_file.name, "text": report_text[:2000] if report_text else "No text extracted"})
                if report_text:
                    gemini_analysis_json = analyze_medical_report_with_gemini(report_text, api_key)
                    if gemini_analysis_json:
                        df_single, patient_info_single = create_structured_dataframe(gemini_analysis_json, uploaded_file.name)
                        if not df_single.empty:
                            all_dfs.append(df_single)
                        if patient_info_single:
                            all_patient_infos_from_pdfs.append(patient_info_single)
                        st.success(f"✅ Analyzed: {uploaded_file.name}")
                    else:
                        st.error(f"⚠️ Failed to get structured analysis from AI for {uploaded_file.name}.")
                else:
                    st.error(f"⚠️ Could not extract text from {uploaded_file.name}.")

        new_data_df = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

        combined_raw_df = pd.DataFrame()
        if upload_mode == "Add new medical reports to an existing Excel/CSV file" and uploaded_excel_file_object:
            try:
                st.write(f"--- Processing existing file: {uploaded_excel_file_object.name} ---")
                if uploaded_excel_file_object.name.endswith('.csv'):
                    existing_pivoted_df = pd.read_csv(uploaded_excel_file_object, index_col='Test_Name')
                else:
                    existing_pivoted_df = pd.read_excel(uploaded_excel_file_object, index_col='Test_Name')

                existing_raw_df = existing_pivoted_df.stack().reset_index(name='Result')
                existing_raw_df.rename(columns={'level_1': 'Test_Date'}, inplace=True)
                existing_raw_df['Test_Name'] = existing_raw_df['Test_Name'].apply(lambda x: standardize_value(x, TEST_NAME_MAPPING, default_case='title'))
                # Use the new date parsing function to handle DD/MM/YYYY format
                existing_raw_df['Test_Date'] = existing_raw_df['Test_Date'].apply(lambda x: format_date_dd_mm_yyyy(parse_date_dd_mm_yyyy(str(x))) if parse_date_dd_mm_yyyy(str(x)) is not None else 'N/A')

                expected_columns = ['Source_Filename', 'Patient_ID', 'Patient_Name', 'Age', 'Gender', 'Test_Date', 'Lab_Name', 'Test_Category', 'Original_Test_Name', 'Test_Name', 'Result', 'Unit', 'Reference_Range', 'Status', 'Processed_Date', 'Result_Numeric', 'Test_Date_dt']
                for col in expected_columns:
                    if col not in existing_raw_df.columns:
                        existing_raw_df[col] = 'N/A'
                
                existing_raw_df['Source_Filename'] = uploaded_excel_file_object.name
                existing_raw_df['Processed_Date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                existing_raw_df['Original_Test_Name'] = existing_raw_df['Test_Name']
                existing_raw_df['Result_Numeric'] = pd.to_numeric(existing_raw_df['Result'], errors='coerce')
                existing_raw_df['Test_Date_dt'] = existing_raw_df['Test_Date'].apply(parse_date_dd_mm_yyyy)
                existing_raw_df = existing_raw_df.reindex(columns=expected_columns)

                if existing_raw_df.empty:
                    st.warning(f"⚠️ No data could be extracted from the existing file: {uploaded_excel_file_object.name}.")
                    combined_raw_df = new_data_df
                else:
                    combined_raw_df = pd.concat([existing_raw_df, new_data_df], ignore_index=True)
                    st.success(f"✅ Processed existing file: {uploaded_excel_file_object.name}")

            except Exception as e:
                st.error(f"Error reading or processing existing Excel/CSV file: {str(e)}")
                combined_raw_df = new_data_df
        else:
            combined_raw_df = new_data_df

        consolidated_info = consolidate_patient_info(all_patient_infos_from_pdfs)
        if 'error' in consolidated_info and consolidated_info['error'] == 'name_mismatch':
            st.warning("⚠️ Different patient names detected in reports!")
            st.write("Found these different names:", ", ".join(consolidated_info['names']))
            st.info("🔄 Automatically resolving name conflicts using smart selection...")
            consolidated_info = create_consolidated_info_with_smart_selection(all_patient_infos_from_pdfs)
            st.success(f"✅ Resolved name conflict. Using: **{consolidated_info.get('name', 'N/A')}**")

        st.session_state.report_df = combined_raw_df
        if not st.session_state.report_df.empty:
            st.session_state.consolidated_patient_info = consolidated_info
            st.session_state.analysis_done = True
            st.balloons()
        else:
            st.warning("No data could be extracted or merged from the provided files.")
            st.session_state.analysis_done = False

if st.session_state.analysis_done and not st.session_state.report_df.empty:
    st.session_state.report_df = unify_test_names(st.session_state.report_df)
    st.session_state.report_df = combine_duplicate_tests(st.session_state.report_df)
    
    st.header("👤 Patient Information")
    p_info = st.session_state.consolidated_patient_info
    if p_info:
        col1, col2, col3 = st.columns(3)
        col1.metric("Name", p_info.get('name', "N/A"))
        col2.metric("Age", p_info.get('age', "N/A"))
        col3.metric("Gender", p_info.get('gender', "N/A"))
    else:
        st.info("No patient information available.")
    st.divider()

    st.header("💬 Health Report Assistant")
    user_query = st.chat_input("How can I help you today?")

    if not st.session_state.chat_history:
        st.markdown("##### 💡 Try asking:")
        example_questions = ["show a general overview of my report", "Explain my blood test results", "Any concerning findings?", "Show my test trends over time"]
        col1, col2 = st.columns(2)
        cols = [col1, col2, col1, col2]
        for i, question in enumerate(example_questions):
            if cols[i].button(question, key=f"example_q_{i}", use_container_width=True):
                user_query = question

    if user_query:
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.spinner("Thinking..."):
            assistant_response = get_chatbot_response(st.session_state.report_df, user_query, st.session_state.chat_history[:-1], api_key)
            st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})

    for i in range(0, len(st.session_state.chat_history), 2):
        if i + 1 < len(st.session_state.chat_history):
            with st.chat_message("user"):
                st.write(st.session_state.chat_history[i]["content"])
            with st.chat_message("assistant", avatar="🤖"):
                st.write(st.session_state.chat_history[i + 1]["content"])
    st.divider()

    st.header("📈 Test Result Visualizations", divider='rainbow')
    viz_container = st.container()
    with viz_container:
        col1, col2 = st.columns([1, 3])
        with col1:
            df_for_viz = st.session_state.report_df.copy()
            df_for_viz = df_for_viz[~df_for_viz['Test_Name'].isin(['N/A', 'UnknownTest', 'Unknown Test']) & df_for_viz['Test_Name'].notna()]
            df_for_viz['Test_Category'] = df_for_viz['Test_Category'].fillna('Other').astype(str)

            if not df_for_viz.empty:
                all_body_parts = set()
                for category in df_for_viz['Test_Category'].unique():
                    all_body_parts.update(TEST_CATEGORY_TO_BODY_PARTS.get(category, ["General"]))
                
                body_part_options = ["-- All Systems --"] + [f"{BODY_PARTS_TO_EMOJI.get(bp, '📊')} {bp}" for bp in sorted(all_body_parts)]
                selected_body_part_display = st.selectbox("Filter by Body System:", options=body_part_options, key="body_part_selector")
                selected_body_part = selected_body_part_display.split(" ", 1)[1] if selected_body_part_display != "-- All Systems --" else None

                if selected_body_part and selected_body_part != "General":
                    relevant_categories = [cat for cat in df_for_viz['Test_Category'].unique() if selected_body_part in TEST_CATEGORY_TO_BODY_PARTS.get(cat, ["General"])]
                    df_for_viz = df_for_viz[df_for_viz['Test_Category'].isin(relevant_categories)]

                category_body_parts = {category: f"{' '.join([BODY_PARTS_TO_EMOJI.get(bp, '📊') for bp in TEST_CATEGORY_TO_BODY_PARTS.get(category, ['General'])])} {category}" for category in df_for_viz['Test_Category'].unique()}
                
                available_categories = sorted(df_for_viz['Test_Category'].unique().tolist())
                if not available_categories:
                    st.info("No test categories found for visualization.")
                else:
                    category_options = ["-- All Categories --"] + [category_body_parts.get(cat, f"📊 {cat}") for cat in available_categories]
                    selected_category_display = st.selectbox("Select Body System to Analyze:", options=category_options, key="category_selector")
                    selected_category = "-- All Categories --" if selected_category_display == "-- All Categories --" else next(cat for cat in available_categories if category_body_parts.get(cat, f"📊 {cat}") == selected_category_display)

                    tests_in_category = sorted(df_for_viz[df_for_viz['Test_Category'] == selected_category]['Test_Name'].unique().tolist()) if selected_category and selected_category != "-- All Categories --" else sorted(df_for_viz['Test_Name'].unique().tolist())
                    
                    if not tests_in_category:
                        st.info(f"No tests found for category: {selected_category}")
                    else:
                        selected_test = st.selectbox("Select a specific test to visualize:", options=["-- Select a test --"] + tests_in_category, key="test_selector_viz")
                        
                        if selected_test and selected_test != "-- Select a test --":
                            test_specific_data = df_for_viz[df_for_viz['Test_Name'] == selected_test]
                            available_dates = sorted(test_specific_data['Test_Date'].unique().tolist(), reverse=True)
                            
                            selected_plot_date = "All Dates"
                            if len(available_dates) > 1:
                                selected_plot_date = st.selectbox("Select Report Date for Plot (or 'All Dates' for trend):", options=["All Dates"] + available_dates, key="date_selector_viz")
                            elif len(available_dates) == 1:
                                selected_plot_date = available_dates[0]

        with col2:
            if 'selected_test' in locals() and selected_test and selected_test != "-- Select a test --":
                plot = generate_test_plot(df_for_viz, selected_test, selected_plot_date)
                if plot:
                    st.plotly_chart(plot, use_container_width=True, config={'displayModeBar': True})
                else:
                    st.info(f"Could not generate plot for {selected_test}. This might be due to non-numeric results or missing reference ranges for the selected date(s).")
            elif 'selected_test' in locals() and selected_test == "-- Select a test --":
                st.markdown("""
                    <div style='text-align: center; padding: 50px; color: #666;'>
                        <h3>📊 Chart Area</h3>
                        <p>Please select a specific test from the dropdown on the left to see its visualization.</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div style='text-align: center; padding: 50px; color: #666;'>
                        <h3>📊 Chart Area</h3>
                        <p>No plottable test results found in the report(s) to visualize.</p>
                    </div>
                """, unsafe_allow_html=True)

        st.divider()

    st.header("📊 Organised Data by Date")
    
    if not st.session_state.report_df.empty:
        try:
            df_with_date_lab = st.session_state.report_df.copy()
            def clean_lab_name(lab_name):
                if pd.isna(lab_name) or lab_name == 'N/A':
                    return 'N/A'
                lab_name = str(lab_name).strip()
                billing_terms = ['bill', 'billing', 'invoice', 'receipt', 'payment', 'charges', 'collection center', 'collection centre']
                parts = []
                for separator in ['-', '|', ',', '(', ')']:
                    if separator in lab_name:
                        parts = lab_name.split(separator)
                        break
                if not parts:
                    parts = [lab_name]
                cleaned_parts = [part.strip() for part in parts if not any(term in part.lower() for term in billing_terms) and len(part.strip()) > 2 and not part.strip().isdigit()]
                return max(cleaned_parts, key=len) if cleaned_parts else lab_name
            
            df_with_date_lab['Lab_Name_Clean'] = df_with_date_lab['Lab_Name'].apply(clean_lab_name)
            df_with_date_lab['Date_Lab'] = df_with_date_lab['Test_Date'] + '_' + df_with_date_lab['Lab_Name_Clean']
            
            organized_df = df_with_date_lab.pivot_table(index=['Test_Category', 'Test_Name'], columns='Date_Lab', values='Result', aggfunc='first').reset_index()
            ref_range_df = df_with_date_lab.pivot_table(index=['Test_Category', 'Test_Name'], columns='Date_Lab', values='Reference_Range', aggfunc='first').reset_index()
            
            date_lab_cols = [col for col in organized_df.columns if col not in ['Test_Category', 'Test_Name']]
            # Sort dates using the proper DD/MM/YYYY parsing
            date_lab_cols_sorted = sorted(date_lab_cols, key=lambda col_name: parse_date_dd_mm_yyyy(col_name.split('_')[0]) or pd.Timestamp.min)
            
            required_cols = ['Test_Category', 'Test_Name'] + date_lab_cols_sorted
            organized_df = organized_df[required_cols]
            ref_range_df = ref_range_df[required_cols]
            
            organized_df = organized_df.sort_values(['Test_Category', 'Test_Name']).reset_index(drop=True)
            ref_range_df = ref_range_df.sort_values(['Test_Category', 'Test_Name']).reset_index(drop=True)
            
            if organized_df.empty or ref_range_df.empty or len(organized_df) != len(ref_range_df) or not organized_df.columns.equals(ref_range_df.columns):
                st.error("Data structure mismatch or no data available to organize.")
            else:
                display_df = organized_df.copy()
                date_row = {'Test_Category': '📅 Date', 'Test_Name': ''}
                lab_row = {'Test_Category': '🏥 Lab', 'Test_Name': ''}
                for date_lab_col in date_lab_cols_sorted:
                    parts = date_lab_col.split('_', 1)
                    date_row[date_lab_col] = parts[0] if len(parts) > 0 else 'N/A'
                    lab_row[date_lab_col] = parts[1] if len(parts) > 1 else 'N/A'
                
                header_rows_df = pd.DataFrame([date_row, lab_row])
                display_df = pd.concat([header_rows_df, display_df], ignore_index=True)
                
                st.write("Your medical test results organised by test category, test name, and date with corresponding lab names.")
                
                def highlight_header_rows(row):
                    if row.name == 0:
                        return ['background-color: #E7E6E6; font-weight: bold; color: #2E5C8F'] * len(row)
                    elif row.name == 1:
                        return ['background-color: #F0F8FF; font-weight: bold; color: #1E7B3E; font-style: italic'] * len(row)
                    return [''] * len(row)
                
                styled_df = display_df.style.apply(highlight_header_rows, axis=1)
                st.dataframe(styled_df, use_container_width=True)

                patient_name_for_file = "".join(c if c.isalnum() else "_" for c in p_info.get('name', 'medical_data'))
                excel_data = create_enhanced_excel_with_trends(organized_df, ref_range_df, date_lab_cols_sorted, p_info)

                organized_csv = display_df.to_csv(index=False).encode('utf-8')

                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="📥 Download Enhanced Excel with Lab Names & Trend Charts",
                        data=excel_data,
                        file_name=f"medical_reports_with_labs_trends_{patient_name_for_file}.xlsx",
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                    )
                with col2:
                    st.download_button(
                        label="📥 Download Organized Data as CSV",
                        data=organized_csv,
                        file_name=f"organized_medical_data_{patient_name_for_file}.csv",
                        mime='text/csv',
                    )
                
                st.info("📊 **Excel Features:**\n- Organized data by category & test\n- Clean lab facility names\n- Reference ranges in trend charts\n- Data table below each chart\n- Embedded trend line charts\n- Summary sheet with patient & lab info\n- Auto-filtering and frozen panes")
                
        except Exception as e:
            st.error(f"Error generating organised data: {str(e)}")
            st.info("Could not create the organised data table. This might happen if there are duplicate test entries for the same date.")
    else:
        st.info("No data available to organise. Please analyze reports first.")

    st.header("🗂️ Extracted Report Data Details")
    with st.expander("View/Hide Raw Extracted Data Table", expanded=False):
        st.dataframe(st.session_state.report_df.drop(columns=['Result_Numeric', 'Test_Date_dt'], errors='ignore'))
        
        csv_export = st.session_state.report_df.to_csv(index=False).encode('utf-8')
        patient_name_for_file = "".join(c if c.isalnum() else "_" for c in p_info.get('name', 'medical_data'))
        st.download_button(
            label="📥 Download All Data as CSV",
            data=csv_export,
            file_name=f"all_medical_reports_{patient_name_for_file}.csv",
            mime='text/csv',
        )
    
    with st.expander("View Raw Text Snippets from PDFs", expanded=False):
        if st.session_state.raw_texts:
            for item in st.session_state.raw_texts:
                st.subheader(f"Snippet from: {item['name']}")
                st.text_area("", item['text'], height=150, disabled=True, key=f"raw_text_{item['name']}")
        else:
            st.info("No raw text snippets available. Analyze reports first.")

elif st.session_state.analysis_done and st.session_state.report_df.empty:
    st.warning("Analysis was run, but no data was extracted or processed. Please check the PDF content and API key.")