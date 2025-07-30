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

# Add the directory containing this script to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir) # Assuming unify_test_names.py is in the parent directory of the script's directory if running from a subdirectory like 'health_app'
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
# If unify_test_names.py is in the same directory as Medical_Project.py, use script_dir instead of parent_dir
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
gemini_model_chat = None # Renamed from gemini_model_summary for clarity

# --- Helper Functions ---
def standardize_value(value, mapping_dict, default_case='title'):
    if not isinstance(value, str):
        return value
    original_value = value.strip()
    standardized_value = original_value
    for pattern, standard_form in mapping_dict.items():
        if pattern.search(original_value):
            standardized_value = standard_form
            break
    if default_case == 'title':
        return standardized_value.title() if standardized_value else standardized_value
    elif default_case == 'original':
        return standardized_value
    elif default_case == 'lower':
        return standardized_value.lower() if standardized_value else standardized_value
    return standardized_value

def extract_text_from_pdf(file_content):
    try:
        pdf_file = io.BytesIO(file_content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page_num, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            if page_text:
                text += f"\n--- Page {page_num+1} ---\n{page_text}" # Add page separator
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
        
        # Reset models to force reinitialization with new key
        gemini_model_extraction = None
        gemini_model_chat = None
        
        genai.configure(api_key=api_key_for_gemini)
        
        # Always create new model instances with the new key
        gemini_model_extraction = genai.GenerativeModel('gemini-2.5-flash')
        gemini_model_chat = genai.GenerativeModel('gemini-2.5-flash')
        
        # Test the connection with a simple generation
        test_response = gemini_model_extraction.generate_content("Test connection")
        if test_response:
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
    The report date or collection date is also critical. Ensure date is in DD-MM-YYYY format if possible. If multiple dates are present (collection, report), prefer collection date.

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
            "date": "Test Date or Report Date (DD-MM-YYYY)"
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
        
        # Enhanced JSON extraction
        match_json_block = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if match_json_block:
            json_str = match_json_block.group(1)
        else:
            # Fallback: try to find the first '{' and last '}'
            start_index = response_text.find('{')
            end_index = response_text.rfind('}')
            if start_index != -1 and end_index != -1 and end_index > start_index:
                json_str = response_text[start_index : end_index+1]
            else: # Fallback if no clear JSON block is found
                st.error("Could not find a clear JSON block in Gemini API response.")
                st.text_area("Gemini API Response (text):", response_text, height=150)
                return None
        
        try:
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
        return pd.DataFrame(), {} # Return empty patient_info_dict

    all_rows = []
    patient_info_dict = ai_results_json.get('patient_info', {})
    
    for test_result in ai_results_json.get('test_results', []):
        raw_test_name = test_result.get('test_name', 'UnknownTest')
        raw_unit = test_result.get('unit', '')
        raw_status = test_result.get('status', '')

        std_test_name = standardize_value(raw_test_name, TEST_NAME_MAPPING, default_case='title')
        std_unit = standardize_value(raw_unit, UNIT_MAPPING, default_case='original')
        std_status = standardize_value(raw_status, STATUS_MAPPING, default_case='title')
        
        # Attempt to parse date from patient_info_dict, could be None
        report_date_str = patient_info_dict.get('date', 'N/A')
        parsed_date = pd.to_datetime(report_date_str, errors='coerce').strftime('%d-%m-%Y') if pd.notna(pd.to_datetime(report_date_str, errors='coerce')) else 'N/A'


        row = {
            'Source_Filename': source_filename,
            'Patient_ID': patient_info_dict.get('patient_id', 'N/A'),
            'Patient_Name': patient_info_dict.get('name', 'N/A'), # Will be refined later
            'Age': patient_info_dict.get('age', 'N/A'),
            'Gender': patient_info_dict.get('gender', 'N/A'),
            'Test_Date': parsed_date,
            'Test_Category': standardize_value(test_result.get('category', 'N/A'), {}, default_case='title'),
            'Original_Test_Name': raw_test_name,
            'Test_Name': std_test_name,
            'Result': test_result.get('result', ''),
            'Unit': std_unit,
            'Reference_Range': test_result.get('reference_range', ''),
            'Status': std_status,
            'Processed_Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        all_rows.append(row)

    if not all_rows:
        return pd.DataFrame(), patient_info_dict

    df = pd.DataFrame(all_rows)
    df['Result_Numeric'] = pd.to_numeric(df['Result'], errors='coerce')
    # Convert Test_Date to datetime for sorting, then back to string if needed, or keep as datetime
    df['Test_Date_dt'] = pd.to_datetime(df['Test_Date'], errors='coerce')
    df = df.sort_values(by=['Test_Date_dt', 'Test_Category', 'Test_Name']).reset_index(drop=True)
    
    return df, patient_info_dict

def normalize_name(name):
    """Normalize a name for comparison by removing titles, extra spaces, and standardizing case."""
    if not name or name == 'N/A':
        return ''
    
    # Convert to title case first and strip
    name = name.strip()
    
    # Remove common titles and prefixes (case insensitive)
    titles = ['mr', 'mrs', 'ms', 'dr', 'prof', 'self', 'mr.', 'mrs.', 'ms.', 'dr.', 'prof.']
    name_lower = name.lower()
    for title in titles:
        name_lower = re.sub(rf'\b{title}\b\.?\s*', '', name_lower)
    
    # Split into words and remove any empty strings
    words = [word for word in name_lower.split() if word]
    
    # Convert to title case and join
    normalized = ' '.join(word.title() for word in words)
    
    return normalized

def are_names_matching(name1, name2):
    """Check if two names match or are variations of the same name."""
    name1 = normalize_name(name1)
    name2 = normalize_name(name2)
    
    if not name1 or not name2:
        return True  # Consider empty/N/A names as matching to handle missing data
        
    name1_parts = set(name1.split())
    name2_parts = set(name2.split())
    
    # If one name is completely contained within another, consider it a match
    if name1_parts.issubset(name2_parts) or name2_parts.issubset(name1_parts):
        return True
        
    # Calculate name parts that match
    matching_parts = name1_parts.intersection(name2_parts)
    total_unique_parts = name1_parts.union(name2_parts)
    
    # If we have at least 2 matching parts (like first and last name)
    # and they make up at least 60% of the total unique parts
    if len(matching_parts) >= 2 and len(matching_parts) / len(total_unique_parts) >= 0.6:
        return True
    
    return False

def consolidate_patient_info(patient_info_list):
    if not patient_info_list:
        return {}

    # First, check for name mismatches and return them if found
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

    ages = [pi.get('age') for pi in patient_info_list if pi.get('age') and pi.get('age') not in ['N/A', '']]
    genders = [pi.get('gender') for pi in patient_info_list if pi.get('gender') and pi.get('gender') not in ['N/A', '']]
    patient_ids = [pi.get('patient_id') for pi in patient_info_list if pi.get('patient_id') and pi.get('patient_id') not in ['N/A', '']]
    dates = [pi.get('date') for pi in patient_info_list if pi.get('date') and pi.get('date') not in ['N/A', '']]

    # Normalize all names first
    normalized_names = [normalize_name(name) for name in names]
    normalized_names = [name for name in normalized_names if name]  # Remove empty strings
    
    final_name = "N/A"
    if normalized_names:
        # Count occurrences of normalized names
        name_counts = Counter(normalized_names)
        most_common_names = name_counts.most_common()
        if most_common_names:
            # Use the most common normalized name, prefer longer names if there's a tie
            max_count = most_common_names[0][1]
            most_frequent_names = [name for name, count in most_common_names if count == max_count]
            final_name = max(most_frequent_names, key=len)# Prefer longer names among the most common ones
            max_len = 0
            best_name_candidate = ""
            for name, count in most_common_names:
                if count == most_common_names[0][1]: # Only consider names with the highest frequency
                    if len(name) > max_len:
                        max_len = len(name)
                        best_name_candidate = name
            final_name = best_name_candidate if best_name_candidate else most_common_names[0][0]


    # Consolidate other info: most frequent valid value
    final_age = Counter(ages).most_common(1)[0][0] if ages else "N/A"
    final_gender = Counter(genders).most_common(1)[0][0] if genders else "N/A"
    final_patient_id = Counter(patient_ids).most_common(1)[0][0] if patient_ids else "N/A"
    
        # For date, might prefer the latest or earliest, or just the most common
    # For now, most common valid date. If dates are datetime objects, this needs adjustment.
    parsed_dates = [pd.to_datetime(d, errors='coerce') for d in dates]
    valid_parsed_dates = [d for d in parsed_dates if pd.notna(d)]
    final_date = max(valid_parsed_dates).strftime('%d-%m-%Y') if valid_parsed_dates else "N/A"
    return {
        'name': final_name,
        'age': final_age,
        'gender': final_gender,
        'patient_id': final_patient_id,
        'date': final_date # This represents the most common/latest date from patient_info blocks, not necessarily all test dates
    }

def get_chatbot_response(report_df_for_prompt, user_question, chat_history_for_prompt, api_key_for_gemini):
    global gemini_model_chat
    if not gemini_model_chat and not init_gemini_models(api_key_for_gemini):
        return "Chatbot model not initialized. API key might be missing or invalid."

    if report_df_for_prompt.empty:
        return "No report data available to answer questions. Please analyze a report first."

    # Convert relevant parts of DataFrame to string for the prompt
    # To manage context window size, we might need to be selective or summarize
    df_string = report_df_for_prompt[['Test_Date', 'Test_Category', 'Test_Name', 'Result', 'Unit', 'Reference_Range', 'Status']].to_string(index=False, max_rows=50) # Limit rows in prompt
    
    # Format chat history for the prompt
    history_context = ""
    for entry in chat_history_for_prompt[-5:]: # Use last 5 interactions for context
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
        return None, None, None # low, high, type (range, less_than, greater_than)
    
    ref_range_str = ref_range_str.strip()
    
    # Try to match a numerical range (e.g., "10.0 - 20.0", "10-20")
    match_range = re.search(r'([\d.]+)\s*-\s*([\d.]+)', ref_range_str)
    if match_range:
        try: return float(match_range.group(1)), float(match_range.group(2)), "range"
        except ValueError: pass

    # Try to match less than (e.g., "< 5.0", "Less than 5")
    match_less_than = re.search(r'(?:<|Less than|upto)\s*([\d.]+)', ref_range_str, re.IGNORECASE)
    if match_less_than:
        try: return None, float(match_less_than.group(1)), "less_than"
        except ValueError: pass

    # Try to match greater than (e.g., "> 10.0", "Greater than 10")
    match_greater_than = re.search(r'(?:>|Greater than|above)\s*([\d.]+)', ref_range_str, re.IGNORECASE)
    if match_greater_than:
        try: return float(match_greater_than.group(1)), None, "greater_than"
        except ValueError: pass
    
    # For qualitative ranges like "Negative", "Non Reactive"
    if ref_range_str.lower() in ["negative", "non reactive", "not detected"]:
        return None, None, "qualitative_normal"
    if ref_range_str.lower() in ["positive", "reactive", "detected"]:
        return None, None, "qualitative_abnormal"
        
    return None, None, None # Default if no pattern matches

def generate_test_plot(df_report, selected_test_name, selected_date=None):
    test_data_for_plot = df_report[df_report['Test_Name'] == selected_test_name]
    
    if selected_date and selected_date != "All Dates":
        test_data_for_plot = test_data_for_plot[test_data_for_plot['Test_Date'] == selected_date]
    
    if test_data_for_plot.empty:
        st.warning(f"No data found for test: {selected_test_name}" + (f" on {selected_date}" if selected_date and selected_date != "All Dates" else ""))
        return None

    # If multiple entries for the same test (e.g. from different files but same date after consolidation, or if "All Dates" is chosen with multiple dates)
    # For now, let's plot the most recent one if multiple dates, or first if same date.
    # A more advanced plot would show a time series if multiple dates are present.
    if "All Dates" == selected_date and len(test_data_for_plot['Test_Date_dt'].unique()) > 1:
         # Create a time series plot
        fig = go.Figure()
        test_data_for_plot = test_data_for_plot.sort_values('Test_Date_dt')
        
        # Check if Result_Numeric can be plotted
        if pd.to_numeric(test_data_for_plot['Result'], errors='coerce').notna().all():
            fig.add_trace(go.Scatter(
                x=test_data_for_plot['Test_Date_dt'], 
                y=test_data_for_plot['Result_Numeric'],
                mode='lines+markers',
                name='Result Trend'
            ))
            
            # Try to plot reference range bands if consistent
            # This is complex if ref range changes over time or is qualitative
            # For simplicity, we'll take the ref range from the latest entry if consistent
            latest_entry = test_data_for_plot.iloc[-1]
            low_ref, high_ref, ref_type = parse_reference_range(latest_entry['Reference_Range'])
            
            if ref_type == "range" and low_ref is not None and high_ref is not None:
                fig.add_trace(go.Scatter(
                    x=test_data_for_plot['Test_Date_dt'], 
                    y=[high_ref] * len(test_data_for_plot),
                    mode='lines', name='Upper Reference', line=dict(dash='dot', color='red')))
                fig.add_trace(go.Scatter(
                    x=test_data_for_plot['Test_Date_dt'], 
                    y=[low_ref] * len(test_data_for_plot),
                    mode='lines', name='Lower Reference', line=dict(dash='dot', color='green'),
                    fill='tonexty', fillcolor='rgba(0,255,0,0.1)')) # Fill between lower and upper
            
            unit = latest_entry['Unit']
            fig.update_layout(
                title_text=f"{selected_test_name} Trend ({unit})",
                xaxis_title="Date", yaxis_title=f"Result ({unit})",
                height=200, margin=dict(l=20, r=20, t=30, b=20),
                showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                autosize=True
            )
            return fig
        else:
            st.info(f"Cannot plot trend for '{selected_test_name}' as some results are non-numeric.")
            return None

    # Single point plot (latest or selected date) - Bullet Chart style
    test_entry = test_data_for_plot.sort_values('Test_Date_dt', ascending=False).iloc[0]
    result_val_numeric = test_entry['Result_Numeric']
    result_val_str = test_entry['Result']
    ref_range_str = test_entry['Reference_Range']
    unit = test_entry['Unit']
    status = test_entry['Status']

    if pd.isna(result_val_numeric):
        st.info(f"Result for '{selected_test_name}' is '{result_val_str}' (non-numeric). Status: {status}")
        # Optionally display qualitative status if needed
        return None

    fig = go.Figure()
    
    low_ref, high_ref, ref_type = parse_reference_range(ref_range_str)
    
    # Define ranges for bullet chart based on ref_type
    # These are indicative ranges. For a true bullet chart, we'd need more defined performance bands.
    plot_min = result_val_numeric
    plot_max = result_val_numeric

    if ref_type == "range": # Normal range defined by low_ref and high_ref
        ranges = [low_ref * 0.8 if low_ref else 0, low_ref, high_ref, high_ref * 1.2 if high_ref else result_val_numeric * 1.5]
        range_colors = ['lightcoral', 'lightgreen', 'lightcoral'] # Low, Normal, High
        actual_ranges_for_bar = [low_ref, high_ref]
        plot_min = min(plot_min, low_ref * 0.75 if low_ref else 0)
        plot_max = max(plot_max, high_ref * 1.25 if high_ref else result_val_numeric * 1.5)

    elif ref_type == "less_than": # Normal is < high_ref
        ranges = [0, high_ref, high_ref * 1.5] # Normal, High
        range_colors = ['lightgreen', 'lightcoral']
        actual_ranges_for_bar = [0, high_ref]
        plot_min = 0
        plot_max = max(plot_max, high_ref * 1.5)

    elif ref_type == "greater_than": # Normal is > low_ref
        ranges = [low_ref * 0.5, low_ref, low_ref*1.5 if result_val_numeric < low_ref else result_val_numeric * 1.2] # Low, Normal
        range_colors = ['lightcoral', 'lightgreen']
        actual_ranges_for_bar = [low_ref, ranges[-1]] # extend normal range visually upwards
        plot_min = min(plot_min, low_ref * 0.5)
        plot_max = max(plot_max, ranges[-1])
    else: # No clear numerical range, fallback to simple bar
        fig.add_trace(go.Bar(
            y=[selected_test_name], x=[result_val_numeric],
            name=f"Your Result: {result_val_str} {unit}", orientation='h',
            marker_color='royalblue', text=f"{result_val_str} {unit}", textposition="auto"
        ))
        fig.update_layout(title_text=f"{selected_test_name} - Result: {result_val_str} {unit} (Status: {status})", height=250)
        return fig

    # Create bullet chart style plot
    fig.add_trace(go.Indicator(
        mode = "number+gauge",
        value = result_val_numeric,
        domain = {'x': [0.1, 1], 'y': [0.3, 0.9]}, # Position gauge in middle
        number = {'suffix': f" {unit}", 'font': {'size': 24}, 'valueformat': '.0f'},
        title = {'text': f"<br><br>{selected_test_name}<br><span style='font-size:0.8em;color:gray'>Reference Range: {ref_range_str}<br>Status: {status}</span>", 'font': {"size": 14}, 'align': "center"},
        gauge = {
            'shape': "bullet",
            'axis': {'range': [min(ranges[0], result_val_numeric*0.8), max(ranges[-1],result_val_numeric*1.2)]}, # Ensure value is visible
            'threshold': {
                'line': {'color': "red", 'width': 2},
                'thickness': 0.75,
                'value': result_val_numeric # This is the marker for the actual value
            },
            'steps': [
                {'range': [ranges[i], ranges[i+1]], 'color': range_colors[i]} for i in range(len(range_colors))
            ],
            'bar': {'color': "rgba(0,0,0,0)"} # Make the main bar transparent, rely on threshold
        }
    ))
    
    fig.update_layout(
        height=150,
        margin=dict(l=10, r=10, t=40, b=10),
        autosize=True
    )
    return fig


# --- Streamlit App UI ---
st.set_page_config(page_title="Medical Report Analyzer", layout="wide")

# Load custom CSS
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.title("⚕️ Medical Report Analyzer & Health Insights")
st.markdown("Upload your medical PDF reports to get structured data, a health summary, and visualizations.")

# --- Gemini API Key from secrets.toml ---
api_key = st.secrets["GEMINI_API_KEY"]
if not init_gemini_models(api_key):
    st.error("🚨 Unable to initialize Gemini models. Please check the server configuration.")

# --- Initialize session state ---
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
if 'report_df' not in st.session_state: # Combined DataFrame from all PDFs
    st.session_state.report_df = pd.DataFrame()
if 'consolidated_patient_info' not in st.session_state:
    st.session_state.consolidated_patient_info = {}
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'raw_texts' not in st.session_state:
    st.session_state.raw_texts = []

# --- New Feature: Upload Mode Selection ---
upload_mode = st.radio(
    "Select what you want to do:",
    [
        "Upload new medical reports",
        "Add new medical reports to an existing Excel/CSV file"
    ],
    index=0,
    key="upload_mode_selector"
)

# Only show the Excel/CSV uploader if the user selects the second option
if upload_mode == "Add new medical reports to an existing Excel/CSV file":
    uploaded_excel_file_object = st.file_uploader(
        "Upload your previously downloaded Excel or CSV file (from this website)",
        type=["csv", "xlsx"],
        accept_multiple_files=False,
        key="excel_uploader"
    )
    if uploaded_excel_file_object is not None:
        st.session_state.uploaded_excel_data = uploaded_excel_file_object.getvalue()
        st.session_state.uploaded_excel_name = uploaded_excel_file_object.name
        st.session_state.uploaded_excel_type = uploaded_excel_file_object.type
else:
    # Remove any previously stored excel data if switching back to PDF-only mode
    st.session_state.pop('uploaded_excel_data', None)
    st.session_state.pop('uploaded_excel_name', None)
    st.session_state.pop('uploaded_excel_type', None)

# --- File Uploader for PDFs ---
uploaded_files = st.file_uploader(
    "📄 Upload Medical Report PDFs (multiple allowed)", 
    type="pdf", 
    accept_multiple_files=True,
    key="pdf_uploader"
)

# Trigger analysis button
if st.button("🔬 Analyze Reports", key="analyze_btn"):
    st.session_state.analysis_done = False
    st.session_state.report_df = pd.DataFrame() # Reset report_df at the start
    st.session_state.consolidated_patient_info = {}
    st.session_state.chat_history = [] # Reset chat on new analysis
    st.session_state.raw_texts = []

    all_dfs = [] # DataFrames from new PDFs (raw format)
    all_patient_infos_from_pdfs = [] # Patient info dicts from new PDFs

    with st.spinner("Processing reports and analyzing with AI... This may take a moment."):
        # --- Process New PDFs ---
        if uploaded_files:
            for i, uploaded_file in enumerate(uploaded_files):
                st.write(f"--- Processing: {uploaded_file.name} ---")
                file_content = uploaded_file.read()
                report_text = extract_text_from_pdf(file_content)
                st.session_state.raw_texts.append({"name": uploaded_file.name, "text": report_text[:2000] if report_text else "No text extracted"}) # Store snippet
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

        # --- Handle Existing Excel/CSV Merge ---
        combined_raw_df = pd.DataFrame()

        # Check if the upload mode is 'Add to existing' AND if the excel data is in session state
        if st.session_state.get('upload_mode_selector') == "Add new medical reports to an existing Excel/CSV file" and 'uploaded_excel_data' in st.session_state:
            try:
                st.write(f"--- Processing existing file from session state: {st.session_state.uploaded_excel_name} ---")
                # Read from the stored data in session state
                if st.session_state.uploaded_excel_name.endswith('.csv'):
                    existing_pivoted_df = pd.read_csv(io.BytesIO(st.session_state.uploaded_excel_data), index_col='Test_Name')
                else:
                    existing_pivoted_df = pd.read_excel(io.BytesIO(st.session_state.uploaded_excel_data), index_col='Test_Name')

                # Unpivot the existing data
                existing_raw_df = existing_pivoted_df.stack().reset_index(name='Result')
                # The column name for dates after stack is usually 'level_1' if index_col was 'Test_Name'
                existing_raw_df.rename(columns={'level_1': 'Test_Date'}, inplace=True)

                # Standardize Test_Name from the existing data using the same mapping as for PDFs
                existing_raw_df['Test_Name'] = existing_raw_df['Test_Name'].apply(
                    lambda x: standardize_value(x, TEST_NAME_MAPPING, default_case='title')
                )

                # Ensure Test_Date is in DD-MM-YYYY format for consistency before merging
                existing_raw_df['Test_Date'] = pd.to_datetime(existing_raw_df['Test_Date'], errors='coerce').dt.strftime('%d-%m-%Y')

                # Add missing columns with default values to match the structure of new_data_df
                # Define the full set of expected columns in the raw DataFrame
                expected_columns = [
                    'Source_Filename', 'Patient_ID', 'Patient_Name', 'Age', 'Gender',
                    'Test_Date', 'Test_Category', 'Original_Test_Name', 'Test_Name',
                    'Result', 'Unit', 'Reference_Range', 'Status', 'Processed_Date',
                    'Result_Numeric', 'Test_Date_dt'
                ]

                # Add columns that exist in expected_columns but not in existing_raw_df
                for col in expected_columns:
                    if col not in existing_raw_df.columns:
                        existing_raw_df[col] = 'N/A' # Default value

                # Fill specific columns with more meaningful defaults where possible
                existing_raw_df['Source_Filename'] = st.session_state.uploaded_excel_name # Use stored name
                existing_raw_df['Processed_Date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                # For existing data, Original_Test_Name might be the same as the standardized one after the above step
                existing_raw_df['Original_Test_Name'] = existing_raw_df['Test_Name']

                # Convert data types for consistency (Result_Numeric and Test_Date_dt)
                existing_raw_df['Result_Numeric'] = pd.to_numeric(existing_raw_df['Result'], errors='coerce')
                existing_raw_df['Test_Date_dt'] = pd.to_datetime(existing_raw_df['Test_Date'], errors='coerce')

                # Reindex to ensure correct column order
                existing_raw_df = existing_raw_df.reindex(columns=expected_columns)

                # Check if existing_raw_df is empty after processing
                if existing_raw_df.empty:
                    st.warning(f"⚠️ No data could be extracted or unpivoted from the existing file: {st.session_state.uploaded_excel_name}.")
                    combined_raw_df = new_data_df # Fallback to only new data
                else:
                    combined_raw_df = existing_raw_df
                    if not new_data_df.empty:
                        # Concatenate existing raw data with new raw data from PDFs
                        combined_raw_df = pd.concat([combined_raw_df, new_data_df], ignore_index=True)

                    st.success(f"✅ Processed existing file from session state: {st.session_state.uploaded_excel_name}")

            except Exception as e:
                st.error(f"Error reading or processing existing Excel/CSV file from session state: {str(e)}")
                st.error(f"Details: {e}") # Print exception details
                # If existing file processing fails, fall back to just new data
                combined_raw_df = new_data_df
        else:
            # If not adding to existing, or if no excel data in session state, just use the new data from PDFs
            combined_raw_df = new_data_df

        # --- Finalize and Store Data ---
        # Initialize proceed_anyway in session state if not present
        if 'proceed_anyway' not in st.session_state:
            st.session_state.proceed_anyway = False
            
        # First check for name compatibility
        consolidated_info = consolidate_patient_info(all_patient_infos_from_pdfs)
        
        # Initialize or update proceed_anyway in session state if not present
        if 'proceed_anyway' not in st.session_state:
            st.session_state.proceed_anyway = False
        
        if 'error' in consolidated_info and consolidated_info['error'] == 'name_mismatch' and not st.session_state.proceed_anyway:
            # Show name mismatch warning with buttons
            st.warning("⚠️ Different patient names detected in reports!")
            st.write("Found these different names:", ", ".join(consolidated_info['names']))
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Reload & Try Again"):
                    st.session_state.proceed_anyway = False
                    st.rerun()
            with col2:
                if st.button("✅ Proceed Anyway"):
                    st.session_state.proceed_anyway = True
                    # Rerun consolidation without name check
                    consolidated_info = {
                        'name': max(consolidated_info['names'], key=len),  # Use longest name
                        'age': all_patient_infos_from_pdfs[0].get('age', 'N/A'),
                        'gender': all_patient_infos_from_pdfs[0].get('gender', 'N/A'),
                        'patient_id': all_patient_infos_from_pdfs[0].get('patient_id', 'N/A'),
                        'date': all_patient_infos_from_pdfs[0].get('date', 'N/A')
                    }
        
        if not ('error' in consolidated_info and consolidated_info['error'] == 'name_mismatch') or st.session_state.proceed_anyway:
            # Process the data whether we're proceeding anyway or there was no name mismatch# Show error popup for name mismatch
            st.error("⚠️ Name Mismatch Detected!")
            name1, name2 = consolidated_info.get('conflicting_names', ['Unknown', 'Unknown'])
            
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.warning(
                    f"The uploaded reports appear to be for different patients:\n\n"
                    f"• {name1}\n"
                    f"• {name2}\n\n"
                    "These names appear to be different people. If this is incorrect, please ensure:\n"
                    "1. All reports belong to the same person\n"
                    "2. Names are spelled consistently across reports\n"
                )
            
            with col2:
                if st.button("🔄 Reload & Try Again"):
                    st.session_state.clear()
                    st.experimental_rerun()
            
            with col3:
                if st.button("✅ Proceed Anyway"):
                    st.session_state.proceed_anyway = True
            
            if not st.session_state.proceed_anyway:
                st.error(
                    "For privacy and accuracy, we cannot combine reports that appear to be for different patients. "
                    "Please verify the reports and try uploading again."
                )
                st.stop()
            else:
                # If proceeding anyway, create a new consolidated info without the error
                consolidated_info = {
                    'name': consolidated_info['names'][0],  # Use the first name
                    'age': next((pi.get('age', 'N/A') for pi in all_patient_infos_from_pdfs if pi.get('age')), 'N/A'),
                    'gender': next((pi.get('gender', 'N/A') for pi in all_patient_infos_from_pdfs if pi.get('gender')), 'N/A'),
                    'patient_id': next((pi.get('patient_id', 'N/A') for pi in all_patient_infos_from_pdfs if pi.get('patient_id')), 'N/A'),
                    'date': next((pi.get('date', 'N/A') for pi in all_patient_infos_from_pdfs if pi.get('date')), 'N/A')
                }
        
        # If names match or we're proceeding anyway, continue with data processing
        st.session_state.report_df = combined_raw_df

        if not st.session_state.report_df.empty:
            st.session_state.consolidated_patient_info = consolidated_info
            st.session_state.analysis_done = True
            st.balloons()
        else:
            st.warning("No data could be extracted or merged from the provided files.")
            st.session_state.analysis_done = False # Reset analysis_done if no data

# --- Utility: Combine duplicate test names and assign most common category ---
def combine_duplicate_tests(df):
    # Normalize test names (already done via standardize_value, but just in case)
    df = df.copy()
    # Group by Test_Name, Test_Category, Test_Date to count occurrences
    group = df.groupby(['Test_Name', 'Test_Category', 'Test_Date']).size().reset_index(name='count')
    # For each Test_Name, get the category with the most test_dates (count unique dates)
    test_cat_counts = (
        df.groupby(['Test_Name', 'Test_Category'])['Test_Date']
        .nunique()
        .reset_index()
        .rename(columns={'Test_Date': 'date_count'})
    )
    # For ties, use the category with the most total rows in the CSV
    test_cat_total = (
        df.groupby(['Test_Name', 'Test_Category']).size().reset_index(name='row_count')
    )
    # Merge counts
    merged = pd.merge(test_cat_counts, test_cat_total, on=['Test_Name', 'Test_Category'])
    # For each Test_Name, pick the best Test_Category
    def pick_category(subdf):
        # First try to pick by most dates
        max_dates = subdf['date_count'].max()
        date_winners = subdf[subdf['date_count'] == max_dates]
        if len(date_winners) == 1:
            return date_winners.iloc[0]['Test_Category']
        # If tie in dates, use most rows
        max_rows = date_winners['row_count'].max()
        row_winners = date_winners[date_winners['row_count'] == max_rows]
        # Return first category alphabetically if still tied
        return row_winners.iloc[0]['Test_Category']
    best_cats = merged.groupby('Test_Name').apply(pick_category).reset_index()
    best_cats.columns = ['Test_Name', 'Best_Test_Category']
    # Map best category back to df
    df = pd.merge(df, best_cats, on='Test_Name', how='left')
    df['Test_Category'] = df['Best_Test_Category']
    df = df.drop(columns=['Best_Test_Category'])
    # Drop duplicate Test_Name/Test_Date rows, keep first (since category is now unified)
    df = df.sort_values(['Test_Name', 'Test_Date', 'Test_Category']).drop_duplicates(['Test_Name', 'Test_Date'])
    df = df.reset_index(drop=True)
    return df

# --- Main content area: Display after analysis ---
if st.session_state.analysis_done and not st.session_state.report_df.empty:
    # Unify similar test names before combining duplicates
    st.session_state.report_df = unify_test_names(st.session_state.report_df)
    # Combine duplicate test names and unify categories before display/organizing
    st.session_state.report_df = combine_duplicate_tests(st.session_state.report_df)
    
    # --- Patient Information Display ---
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

    # --- Chatbot Interface ---
    st.header("💬 Health Report Assistant")

    # Create a link-like icon in the header
    st.markdown('<div style="text-align: right; margin-top: -50px; color: #4a4a4a;">🔗</div>', unsafe_allow_html=True)

    # Chat input at the top
    user_query = st.chat_input("How can I help you today?")

    # Create a container for the chat interface with dark theme
    chat_container = st.container()

    # Example questions with improved styling - show only if no chat history
    if not st.session_state.chat_history:
        st.markdown("##### 💡 Try asking:")
        
        # Define the example questions in a 2x2 grid
        example_questions = [
            "show a general overview of my report",
            "Explain my blood test results",
            "Any concerning findings?",
            "Show my test trends over time"
        ]
        
        col1, col2 = st.columns(2)
        cols = [col1, col2, col1, col2]
        
        # Display questions in a grid with improved styling
        for i, question in enumerate(example_questions):
            if cols[i].button(
                question,
                key=f"example_q_{i}",
                use_container_width=True,
            ):
                user_query = question

    # Handle user query
    if user_query:
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        
        # Process the query and get response
        with st.spinner("Thinking..."):
            assistant_response = get_chatbot_response(
                st.session_state.report_df,
                user_query,
                st.session_state.chat_history[:-1],
                api_key
            )
            st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})

    # Display chat history in a Q&A format
    for i in range(0, len(st.session_state.chat_history), 2):
        if i + 1 < len(st.session_state.chat_history):
            # Question
            with st.chat_message("user"):
                st.write(st.session_state.chat_history[i]["content"])
            # Answer
            with st.chat_message("assistant", avatar="🤖"):
                st.write(st.session_state.chat_history[i + 1]["content"])

    st.divider()

    # --- Visualizations Section ---
    st.header("📈 Test Result Visualizations", divider='rainbow')
    
    # Create containers for better organization
    viz_container = st.container()
    with viz_container:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            df_for_viz = st.session_state.report_df.copy()
            df_for_viz = df_for_viz[~df_for_viz['Test_Name'].isin(['N/A', 'UnknownTest', 'Unknown Test']) & df_for_viz['Test_Name'].notna()]

            # Ensure all Test_Category values are strings and handle NaN/None
            df_for_viz['Test_Category'] = df_for_viz['Test_Category'].fillna('Other').astype(str)

            if not df_for_viz.empty:
                # Get all unique body parts from all categories
                all_body_parts = set()
                for category in df_for_viz['Test_Category'].unique():
                    body_parts = TEST_CATEGORY_TO_BODY_PARTS.get(category, ["General"])
                    all_body_parts.update(body_parts)
                
                # Create the body part to emoji mapping for display
                body_part_options = ["-- All Systems --"] + [
                    f"{BODY_PARTS_TO_EMOJI.get(bp, '📊')} {bp}" 
                    for bp in sorted(all_body_parts)
                ]

                selected_body_part_display = st.selectbox(
                    "Filter by Body System:",
                    options=body_part_options,
                    key="body_part_selector"
                )

                # Extract the actual body part name
                selected_body_part = selected_body_part_display.split(" ", 1)[1] if selected_body_part_display != "-- All Systems --" else None

                # Filter categories based on selected body part
                if selected_body_part and selected_body_part != "General":
                    relevant_categories = [
                        cat for cat in df_for_viz['Test_Category'].unique()
                        if selected_body_part in TEST_CATEGORY_TO_BODY_PARTS.get(cat, ["General"])
                    ]
                    df_for_viz = df_for_viz[df_for_viz['Test_Category'].isin(relevant_categories)]

                # Create a mapping of categories to body parts for display
                category_body_parts = {}
                for category in df_for_viz['Test_Category'].unique():
                    body_parts = TEST_CATEGORY_TO_BODY_PARTS.get(category, ["General"])
                    emojis = [BODY_PARTS_TO_EMOJI.get(bp, "📊") for bp in body_parts]
                    category_body_parts[category] = f"{' '.join(emojis)} {category}"

        # 1. Dropdown for Body System/Category with emojis
        available_categories = sorted(df_for_viz['Test_Category'].unique().tolist())
        if not available_categories:
            st.info("No test categories found for visualization.")
        else:
            category_options = ["-- All Categories --"] + [
                category_body_parts.get(cat, f"📊 {cat}") 
                for cat in available_categories
            ]
            
            selected_category_display = st.selectbox(
                "Select Body System to Analyze:", 
                options=category_options,
                key="category_selector"
            )
            
            # Extract actual category name from display name
            selected_category = "-- All Categories --" if selected_category_display == "-- All Categories --" else next(
                cat for cat in available_categories 
                if category_body_parts.get(cat, f"📊 {cat}") == selected_category_display
            )

            # 2. Dropdown for Test Name (filtered by category)
            if selected_category and selected_category != "-- All Categories --":
                tests_in_category = sorted(df_for_viz[df_for_viz['Test_Category'] == selected_category]['Test_Name'].unique().tolist())
            else: # All categories selected
                tests_in_category = sorted(df_for_viz['Test_Name'].unique().tolist())
            
            if not tests_in_category:
                st.info(f"No tests found for category: {selected_category}")
            else:
                selected_test = st.selectbox(
                    "Select a specific test to visualize:", 
                    options=["-- Select a test --"] + tests_in_category, 
                    key="test_selector_viz"
                )
                
                # 3. Dropdown for Test Date (if multiple dates exist for the selected test)
                if selected_test and selected_test != "-- Select a test --":
                    test_specific_data = df_for_viz[df_for_viz['Test_Name'] == selected_test]
                    available_dates = sorted(test_specific_data['Test_Date'].unique().tolist(), reverse=True)
                    
                    selected_plot_date = "All Dates" # Default
                    if len(available_dates) > 1:
                        selected_plot_date = st.selectbox(
                            "Select Report Date for Plot (or 'All Dates' for trend):",
                            options=["All Dates"] + available_dates,
                            key="date_selector_viz"
                        )
                    elif len(available_dates) == 1:
                        selected_plot_date = available_dates[0]
                        # st.caption(f"Displaying data for {selected_test} from report dated {selected_plot_date}")


                    plot = generate_test_plot(df_for_viz, selected_test, selected_plot_date)
                    if plot:
                        st.plotly_chart(plot, use_container_width=True)
                    elif selected_test != "-- Select a test --": # If plot is None but a test was selected
                        st.info(f"Could not generate plot for {selected_test}. This might be due to non-numeric results or missing reference ranges for the selected date(s).")

                elif selected_test == "-- Select a test --":
                    st.info("Please select a specific test from the dropdown to see its visualization.")
                else:
                    st.info("No plottable test results found in the report(s) to visualize.")
            
                st.divider()

    # --- Organised Data Section ---
    # --- Organised Data Section with Enhanced Excel Export ---
    st.header("📊 Organised Data by Date")
    if not st.session_state.report_df.empty:
        try:
            # Create pivot table with Test_Category as primary index, then Test_Name
            organized_df = st.session_state.report_df.pivot_table(
                index=['Test_Category', 'Test_Name'],
                columns='Test_Date',
                values='Result',
                aggfunc='first'
            )
            organized_df = organized_df.reset_index()
            
            # Reorder columns: Test_Category, Test_Name, then all dates (sorted chronologically)
            date_cols = [col for col in organized_df.columns if col not in ['Test_Category', 'Test_Name']]
            # Sort date columns chronologically
            try:
                date_cols_sorted = sorted(date_cols, key=lambda x: pd.to_datetime(x, format='%d-%m-%Y', errors='coerce'))
            except:
                date_cols_sorted = sorted(date_cols)  # Fallback to alphabetical sort
                
            organized_df = organized_df[['Test_Category', 'Test_Name'] + date_cols_sorted]
            
            # Sort by Test_Category, then Test_Name
            organized_df = organized_df.sort_values(['Test_Category', 'Test_Name']).reset_index(drop=True)
            
            # Display the organized data in Streamlit
            st.write("Download your medical test results organised by test category, test name, and date (columns) with embedded trend charts in Excel.")
            st.dataframe(organized_df, use_container_width=True)

            # Get patient name for filename
            p_info = st.session_state.consolidated_patient_info
            patient_name_for_file = "".join(c if c.isalnum() else "_" for c in p_info.get('name', 'medical_data'))

            # Create enhanced Excel file with trend charts
            output_excel = io.BytesIO()
            with pd.ExcelWriter(output_excel, engine='xlsxwriter') as writer:
                # Write the main data
                organized_df.to_excel(writer, index=False, sheet_name='Medical Data with Trends', startrow=1)
                
                workbook = writer.book
                worksheet = writer.sheets['Medical Data with Trends']
                
                # Add title
                title_format = workbook.add_format({
                    'bold': True,
                    'font_size': 16,
                    'align': 'center',
                    'bg_color': '#4472C4',
                    'font_color': 'white'
                })
                worksheet.merge_range('A1:' + chr(65 + len(organized_df.columns) - 1) + '1', 
                                    'Medical Test Results - Organized by Date with Trends', title_format)
                
                # Format headers
                header_format = workbook.add_format({
                    'bold': True,
                    'bg_color': '#D9E2F3',
                    'border': 1,
                    'align': 'center',
                    'valign': 'vcenter'
                })
                
                # Apply header formatting
                for col_num, value in enumerate(organized_df.columns.values):
                    worksheet.write(1, col_num, value, header_format)
                
                # Format data cells
                data_format = workbook.add_format({
                    'border': 1,
                    'align': 'center',
                    'valign': 'vcenter'
                })
                
                numeric_format = workbook.add_format({
                    'border': 1,
                    'align': 'center',
                    'valign': 'vcenter',
                    'num_format': '0.00'
                })
                
                # Apply data formatting and convert numeric values
                for row_num in range(len(organized_df)):
                    for col_num in range(len(organized_df.columns)):
                        value = organized_df.iloc[row_num, col_num]
                        
                        # Skip Test_Category and Test_Name columns for numeric conversion
                        if col_num < 2:
                            worksheet.write(row_num + 2, col_num, value, data_format)
                        else:
                            # Try to convert to float for date columns
                            try:
                                if pd.notna(value) and str(value).strip() != '':
                                    float_val = float(value)
                                    worksheet.write(row_num + 2, col_num, float_val, numeric_format)
                                else:
                                    worksheet.write(row_num + 2, col_num, value if pd.notna(value) else '', data_format)
                            except (ValueError, TypeError):
                                # Keep as string if not numeric
                                worksheet.write(row_num + 2, col_num, str(value) if pd.notna(value) else '', data_format)
                
                # Set row heights for better chart visibility
                # Set header row height
                worksheet.set_row(1, 35)
                
                # Set data row heights to accommodate much larger charts (250 pixels = about 187.5 points)
                for row_num in range(len(organized_df)):
                    worksheet.set_row(row_num + 2, 187.5)  # Much larger row height for bigger charts
                
                # Add trend charts for each test that has numeric data
                chart_col = len(organized_df.columns)  # Column after the last data column
                chart_row_start = 2  # Start after headers
                
                # Add "Trends Chart" header
                worksheet.write(1, chart_col, "Trends Chart", header_format)
                
                for row_num in range(len(organized_df)):
                    test_category = organized_df.iloc[row_num, 0]
                    test_name = organized_df.iloc[row_num, 1]
                    
                    # Get numeric values for this row (excluding Test_Category and Test_Name)
                    row_values = []
                    date_labels = []
                    
                    for col_idx, col_name in enumerate(date_cols_sorted):
                        value = organized_df.iloc[row_num, col_idx + 2]  # +2 to skip Test_Category and Test_Name
                        if pd.notna(value) and str(value).strip() != '':
                            try:
                                float_val = float(value)
                                row_values.append(float_val)
                                date_labels.append(col_name)
                            except (ValueError, TypeError):
                                continue
                    
                    # Create chart only if we have at least 2 numeric values
                    if len(row_values) >= 2:
                        # Create a line chart
                        chart = workbook.add_chart({'type': 'line'})
                        
                        # Add the data series with thicker line and bigger markers
                        chart.add_series({
                            'name': f'{test_name}',
                            'categories': [worksheet.name, row_num + 2, 2, row_num + 2, len(date_cols_sorted) + 1],
                            'values': [worksheet.name, row_num + 2, 2, row_num + 2, len(date_cols_sorted) + 1],
                            'line': {'color': '#4472C4', 'width': 3},  # Thicker line
                            'marker': {'type': 'circle', 'size': 8, 'border': {'color': '#4472C4', 'width': 2}, 'fill': {'color': '#4472C4'}},  # Bigger markers
                        })
                        
                        # Configure chart
                        chart.set_title({
                            'name': f'{test_name} Trend',
                            'name_font': {'size': 12, 'bold': True}  # Bigger title font
                        })
                        
                        chart.set_x_axis({
                            'name': 'Date',
                            'name_font': {'size': 11, 'bold': True},  # Bigger axis label font
                            'num_font': {'size': 9, 'rotation': 45}  # Bigger axis values font
                        })
                        
                        chart.set_y_axis({
                            'name': 'Value',
                            'name_font': {'size': 11, 'bold': True},  # Bigger axis label font
                            'num_font': {'size': 9}  # Bigger axis values font
                        })
                        
                        chart.set_legend({'none': True})
                        chart.set_size({'width': 450, 'height': 180})  # Much larger chart size
                        
                        # Insert chart in the trends column with proper positioning
                        # Position chart slightly inset from cell boundaries for better appearance
                        worksheet.insert_chart(row_num + 2, chart_col, chart, {
                            'x_offset': 5,
                            'y_offset': 5
                        })
                        
                    else:
                        # If no numeric trend available, write "No trend data"
                        worksheet.write(row_num + 2, chart_col, "No numeric trend data", data_format)
                
                # Adjust column widths
                worksheet.set_column('A:A', 20)  # Test Category
                worksheet.set_column('B:B', 25)  # Test Name
                for i, col in enumerate(date_cols_sorted):
                    worksheet.set_column(i + 2, i + 2, 12)  # Date columns
                worksheet.set_column(chart_col, chart_col, 60)  # Much wider trends column for larger charts
                
                # Add autofilter
                worksheet.autofilter(1, 0, len(organized_df) + 1, len(organized_df.columns) - 1)
                
                # Freeze panes for better navigation
                worksheet.freeze_panes(2, 2)
                
                # Add summary sheet
                summary_sheet = workbook.add_worksheet('Summary')
                
                # Summary title
                summary_sheet.merge_range('A1:D1', 'Medical Report Summary', title_format)
                
                # Patient info
                info_format = workbook.add_format({'bold': True, 'bg_color': '#E7E6E6'})
                summary_sheet.write('A3', 'Patient Information:', info_format)
                summary_sheet.write('A4', f"Name: {p_info.get('name', 'N/A')}")
                summary_sheet.write('A5', f"Age: {p_info.get('age', 'N/A')}")
                summary_sheet.write('A6', f"Gender: {p_info.get('gender', 'N/A')}")
                summary_sheet.write('A7', f"Patient ID: {p_info.get('patient_id', 'N/A')}")
                
                # Report statistics
                summary_sheet.write('A9', 'Report Statistics:', info_format)
                summary_sheet.write('A10', f"Total Test Categories: {organized_df['Test_Category'].nunique()}")
                summary_sheet.write('A11', f"Total Tests: {len(organized_df)}")
                summary_sheet.write('A12', f"Date Range: {min(date_cols_sorted)} to {max(date_cols_sorted)}")
                summary_sheet.write('A13', f"Generated on: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}")
                
                # Test categories breakdown
                summary_sheet.write('A15', 'Test Categories:', info_format)
                categories = organized_df['Test_Category'].value_counts()
                for i, (category, count) in enumerate(categories.items()):
                    summary_sheet.write(f'A{16+i}', f"• {category}: {count} tests")

            excel_data = output_excel.getvalue()

            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="📥 Download Enhanced Excel with Trend Charts",
                    data=excel_data,
                    file_name=f"medical_reports_with_trends_{patient_name_for_file}.xlsx",
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                )
            
            with col2:
                st.info("📊 **Excel Features:**\n- Organized data by category & test\n- Numeric values converted to numbers\n- Embedded trend line charts\n- Summary sheet with patient info\n- Auto-filtering and frozen panes")
                
        except Exception as e:
            st.error(f"Error generating organised data: {str(e)}")
            st.info("Could not create the organised data table. This might happen if there are duplicate test entries for the same date.")
    else:
        st.info("No data available to organise. Please analyze reports first.")

    # --- Extracted Data Section (Moved to Bottom and Collapsible) ---
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
    st.warning("Analysis was run, but no data was extracted or processed from any PDF. Cannot show details. Please check the PDF content and API key.")

