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
from test_category_mapping import TEST_CATEGORY_TO_BODY_PARTS, BODY_PARTS_TO_EMOJI, TEST_NAME_MAPPING, UNIT_MAPPING, STATUS_MAPPING
from unify_test_names import unify_test_names
from Helper_Functions import *
import sys
import os
from collections import Counter



def format_date_dd_mm_yyyy(date_obj):
    """Format datetime object to DD-MM-YYYY string"""
    if pd.isna(date_obj) or date_obj is None:
        return 'N/A'
    return date_obj.strftime('%d-%m-%Y')

def extract_patient_info_from_excel_filename(filename):
    """Try to extract patient info from Excel filename if possible"""
    # This is a fallback - try to extract patient name from filename
    name_match = re.search(r'medical_reports?_.*?_([^_]+(?:_[^_]+)*)', filename.lower())
    if name_match:
        # Replace underscores with spaces and title case
        extracted_name = name_match.group(1).replace('_', ' ').title()
        return extracted_name
    return "N/A"



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


def extract_patient_info_from_normalized_data(df):
    """Extract patient info from normalized dataframe"""
    if df.empty:
        return {}
    
    # Get the most common/recent patient info
    patient_info = {}
    
    # Get most common values
    if 'Patient_Name' in df.columns:
        names = df['Patient_Name'].dropna()
        names = names[names != 'N/A']
        patient_info['name'] = names.mode().iloc[0] if not names.empty else 'N/A'
    
    if 'Age' in df.columns:
        ages = df['Age'].dropna()
        ages = ages[ages != 'N/A']
        patient_info['age'] = ages.mode().iloc[0] if not ages.empty else 'N/A'
    
    if 'Gender' in df.columns:
        genders = df['Gender'].dropna()
        genders = genders[genders != 'N/A']
        patient_info['gender'] = genders.mode().iloc[0] if not genders.empty else 'N/A'
    
    if 'Patient_ID' in df.columns:
        ids = df['Patient_ID'].dropna()
        ids = ids[ids != 'N/A']
        patient_info['patient_id'] = ids.mode().iloc[0] if not ids.empty else 'N/A'
    
    if 'Lab_Name' in df.columns:
        labs = df['Lab_Name'].dropna()
        labs = labs[labs != 'N/A']
        patient_info['lab_name'] = labs.mode().iloc[0] if not labs.empty else 'N/A'
    
    # Get most recent date
    if 'Test_Date' in df.columns:
        dates = df['Test_Date'].dropna()
        dates = dates[dates != 'N/A']
        if not dates.empty:
            # Parse dates and get the most recent one
            parsed_dates = [parse_date_dd_mm_yyyy(d) for d in dates]
            valid_dates = [d for d in parsed_dates if d is not None]
            if valid_dates:
                most_recent = max(valid_dates)
                patient_info['date'] = format_date_dd_mm_yyyy(most_recent)
            else:
                patient_info['date'] = 'N/A'
        else:
            patient_info['date'] = 'N/A'
    
    return patient_info
def auto_detect_and_process(df, filename, new_patient_info_list):
    """Auto-detect the format and process accordingly"""
    
    # Check if it looks like a simple test results table
    if len(df.columns) >= 2:
        # Assume first column is test name, others are dates/results
        test_col = df.columns[0]
        result_cols = df.columns[1:]
        
        # Try to detect if columns are dates
        date_like_cols = []
        for col in result_cols:
            if parse_date_dd_mm_yyyy(str(col)) is not None:
                date_like_cols.append(col)
        
        if date_like_cols:
            # Treat as a simple pivoted format
            print(f"Detected simple pivoted format with {len(date_like_cols)} date columns")
            
            # Convert to our standard pivoted format
            simple_df = df.copy()
            simple_df['Test_Category'] = 'General'  # Default category
            simple_df = simple_df.rename(columns={test_col: 'Test_Name'})
            
            # Reorder columns
            cols = ['Test_Category', 'Test_Name'] + [col for col in simple_df.columns if col not in ['Test_Category', 'Test_Name']]
            simple_df = simple_df[cols]
            
            return process_pivoted_excel_data(simple_df, filename, new_patient_info_list)
    
    print("Could not detect format, returning empty DataFrame")
    return pd.DataFrame(), {}



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

# Replace the existing function in your Medical_Project.py file with this enhanced version
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
        
        # =================== NEW CHARTS TAB ===================
        # Create a new worksheet for individual test charts
        chart_worksheet = workbook.add_worksheet('Test Trend Charts')
        
        # Add title to charts sheet
        chart_worksheet.merge_range('A1:H1', 'Individual Test Trend Charts', title_format)
        chart_worksheet.write('A2', f'Patient: {patient_info.get("name", "N/A")} | Age: {patient_info.get("age", "N/A")} | Gender: {patient_info.get("gender", "N/A")}', 
                             workbook.add_format({'font_size': 12, 'italic': True}))
        
        # Prepare data for charts
        chart_row = 4  # Starting row for charts
        charts_per_row = 2  # Number of charts per row
        chart_width = 480  # Width of each chart in pixels
        chart_height = 300  # Height of each chart in pixels
        chart_spacing_x = 8  # Column spacing between charts (in Excel columns)
        chart_spacing_y = 20  # Row spacing between charts (in Excel rows)
        
        chart_count = 0
        
        # Extract dates for x-axis (parse them properly for sorting)
        dates_for_charts = []
        for date_lab_col in date_lab_cols_sorted:
            date_part = date_lab_col.split('_')[0]
            dates_for_charts.append(date_part)
        
        # Group tests by category for better organization
        tests_by_category = {}
        for idx, row in organized_df.iterrows():
            category = row['Test_Category']
            test_name = row['Test_Name']
            if category not in tests_by_category:
                tests_by_category[category] = []
            tests_by_category[category].append((idx, test_name))
        
        # Create charts for each test that has numeric data
        for category, tests in tests_by_category.items():
            for test_idx, test_name in tests:
                # Get the data for this test
                test_data = []
                has_numeric_data = False
                
                for col_idx, date_lab_col in enumerate(date_lab_cols_sorted):
                    value = organized_df.iloc[test_idx, col_idx + 2]  # +2 to skip category and test name columns
                    try:
                        if pd.notna(value) and str(value).strip() != '':
                            numeric_value = float(value)
                            test_data.append(numeric_value)
                            has_numeric_data = True
                        else:
                            test_data.append(None)
                    except (ValueError, TypeError):
                        test_data.append(None)
                
                # Only create chart if there's numeric data and more than one data point
                valid_data_points = [x for x in test_data if x is not None]
                if has_numeric_data and len(valid_data_points) > 1:
                    # Calculate chart position
                    row_position = chart_count // charts_per_row
                    col_position = chart_count % charts_per_row
                    
                    chart_start_row = chart_row + (row_position * chart_spacing_y)
                    chart_start_col = col_position * chart_spacing_x
                    
                    # Create a line chart
                    chart = workbook.add_chart({'type': 'line'})
                    
                    # Create a temporary data table on the chart sheet for this specific test
                    data_start_row = chart_start_row + 16  # Place data below the chart
                    
                    # Write dates header
                    chart_worksheet.write(data_start_row, chart_start_col, 'Date', 
                                        workbook.add_format({'bold': True, 'bg_color': '#D9E2F3'}))
                    chart_worksheet.write(data_start_row + 1, chart_start_col, test_name, 
                                        workbook.add_format({'bold': True, 'bg_color': '#D9E2F3'}))
                    
                    # Write dates and values - only write actual data points (no empty cells)
                    valid_dates = []
                    valid_values = []
                    date_col_positions = []
                    
                    for i, (date, value) in enumerate(zip(dates_for_charts, test_data)):
                        if value is not None:  # Only include dates with actual values
                            valid_dates.append(date)
                            valid_values.append(value)
                            date_col_positions.append(chart_start_col + 1 + len(valid_dates) - 1)
                    
                    # Write only the valid dates and values (no gaps)
                    for i, (date, value) in enumerate(zip(valid_dates, valid_values)):
                        chart_worksheet.write(data_start_row, chart_start_col + 1 + i, date)
                        chart_worksheet.write_number(data_start_row + 1, chart_start_col + 1 + i, value)
                    
                    # Add data series to chart - only for valid data points
                    chart.add_series({
                        'name': test_name,
                        'categories': ['Test Trend Charts', data_start_row, chart_start_col + 1, 
                                     data_start_row, chart_start_col + len(valid_dates)],
                        'values': ['Test Trend Charts', data_start_row + 1, chart_start_col + 1, 
                                 data_start_row + 1, chart_start_col + len(valid_dates)],
                        'line': {'color': '#4472C4', 'width': 2},
                        'marker': {'type': 'circle', 'size': 6, 'border': {'color': '#4472C4'}, 'fill': {'color': '#4472C4'}},
                    })
                    
                    # Get reference range for this test and parse it
                    ref_range = ''
                    low_ref = None
                    high_ref = None
                    ref_type = None
                    
                    try:
                        for col_idx, col_name in enumerate(date_lab_cols_sorted):
                            if col_idx + 2 < len(ref_range_df.columns):
                                ref_value = ref_range_df.iloc[test_idx, col_idx + 2]
                                if pd.notna(ref_value) and str(ref_value).strip() != '' and str(ref_value) != 'N/A':
                                    ref_range = str(ref_value)
                                    break
                    except:
                        pass
                    
                    # Parse reference range to extract numeric values
                    if ref_range:
                        import re
                        # Try to parse range like "13.0 - 17.0" or "13.0-17.0"
                        match_range = re.search(r'([\d.]+)\s*-\s*([\d.]+)', ref_range)
                        if match_range:
                            try:
                                low_ref = float(match_range.group(1))
                                high_ref = float(match_range.group(2))
                                ref_type = "range"
                            except ValueError:
                                pass
                        else:
                            # Try to parse "< 5.0" or "Less than 5.0"
                            match_less_than = re.search(r'(?:<|Less than|upto)\s*([\d.]+)', ref_range, re.IGNORECASE)
                            if match_less_than:
                                try:
                                    high_ref = float(match_less_than.group(1))
                                    ref_type = "less_than"
                                except ValueError:
                                    pass
                            else:
                                # Try to parse "> 5.0" or "Greater than 5.0"
                                match_greater_than = re.search(r'(?:>|Greater than|above)\s*([\d.]+)', ref_range, re.IGNORECASE)
                                if match_greater_than:
                                    try:
                                        low_ref = float(match_greater_than.group(1))
                                        ref_type = "greater_than"
                                    except ValueError:
                                        pass
                    
                    # Add reference range lines if we have valid numeric ranges
                    if ref_type == "range" and low_ref is not None and high_ref is not None:
                        # Create data for reference lines (same valid dates, constant values)
                        ref_low_row = data_start_row + 2
                        ref_high_row = data_start_row + 3
                        
                        # Write reference line labels
                        chart_worksheet.write(ref_low_row, chart_start_col, 'Lower Limit', 
                                            workbook.add_format({'bold': True, 'font_color': '#228B22'}))
                        chart_worksheet.write(ref_high_row, chart_start_col, 'Upper Limit', 
                                            workbook.add_format({'bold': True, 'font_color': '#DC143C'}))
                        
                        # Write reference values ONLY for valid dates (same pattern as main data)
                        for i in range(len(valid_dates)):
                            chart_worksheet.write_number(ref_low_row, chart_start_col + 1 + i, low_ref)
                            chart_worksheet.write_number(ref_high_row, chart_start_col + 1 + i, high_ref)
                        
                        # Add lower reference line series
                        chart.add_series({
                            'name': f'Lower Limit ({low_ref})',
                            'categories': ['Test Trend Charts', data_start_row, chart_start_col + 1, 
                                         data_start_row, chart_start_col + len(valid_dates)],
                            'values': ['Test Trend Charts', ref_low_row, chart_start_col + 1, 
                                     ref_low_row, chart_start_col + len(valid_dates)],
                            'line': {'color': '#228B22', 'width': 2, 'dash_type': 'dash'},
                            'marker': {'type': 'none'},
                        })
                        
                        # Add upper reference line series
                        chart.add_series({
                            'name': f'Upper Limit ({high_ref})',
                            'categories': ['Test Trend Charts', data_start_row, chart_start_col + 1, 
                                         data_start_row, chart_start_col + len(valid_dates)],
                            'values': ['Test Trend Charts', ref_high_row, chart_start_col + 1, 
                                     ref_high_row, chart_start_col + len(valid_dates)],
                            'line': {'color': '#DC143C', 'width': 2, 'dash_type': 'dash'},
                            'marker': {'type': 'none'},
                        })
                        
                    elif ref_type == "less_than" and high_ref is not None:
                        # Add only upper limit line
                        ref_high_row = data_start_row + 2
                        
                        chart_worksheet.write(ref_high_row, chart_start_col, f'Upper Limit (<{high_ref})', 
                                            workbook.add_format({'bold': True, 'font_color': '#DC143C'}))
                        
                        # Write reference values ONLY for valid dates
                        for i in range(len(valid_dates)):
                            chart_worksheet.write_number(ref_high_row, chart_start_col + 1 + i, high_ref)
                        
                        chart.add_series({
                            'name': f'Upper Limit (<{high_ref})',
                            'categories': ['Test Trend Charts', data_start_row, chart_start_col + 1, 
                                         data_start_row, chart_start_col + len(valid_dates)],
                            'values': ['Test Trend Charts', ref_high_row, chart_start_col + 1, 
                                     ref_high_row, chart_start_col + len(valid_dates)],
                            'line': {'color': '#DC143C', 'width': 2, 'dash_type': 'dash'},
                            'marker': {'type': 'none'},
                        })
                        
                    elif ref_type == "greater_than" and low_ref is not None:
                        # Add only lower limit line
                        ref_low_row = data_start_row + 2
                        
                        chart_worksheet.write(ref_low_row, chart_start_col, f'Lower Limit (>{low_ref})', 
                                            workbook.add_format({'bold': True, 'font_color': '#228B22'}))
                        
                        # Write reference values ONLY for valid dates
                        for i in range(len(valid_dates)):
                            chart_worksheet.write_number(ref_low_row, chart_start_col + 1 + i, low_ref)
                        
                        chart.add_series({
                            'name': f'Lower Limit (>{low_ref})',
                            'categories': ['Test Trend Charts', data_start_row, chart_start_col + 1, 
                                         data_start_row, chart_start_col + len(valid_dates)],
                            'values': ['Test Trend Charts', ref_low_row, chart_start_col + 1, 
                                     ref_low_row, chart_start_col + len(valid_dates)],
                            'line': {'color': '#228B22', 'width': 2, 'dash_type': 'dash'},
                            'marker': {'type': 'none'},
                        })
                    
                    # Adjust Y-axis range to include reference ranges
                    min_val = min(valid_data_points)
                    max_val = max(valid_data_points)
                    
                    # Include reference range values in axis calculation
                    all_values = valid_data_points[:]
                    if low_ref is not None:
                        all_values.append(low_ref)
                    if high_ref is not None:
                        all_values.append(high_ref)
                    
                    final_min = min(all_values)
                    final_max = max(all_values)
                    value_range = final_max - final_min if final_max != final_min else final_max * 0.1
                    y_min = final_min - (value_range * 0.15) if value_range > 0 else final_min * 0.85
                    y_max = final_max + (value_range * 0.15) if value_range > 0 else final_max * 1.15
                    
                    # Configure chart
                    chart.set_title({
                        'name': f'{test_name}\n({category})',
                        'name_font': {'size': 11, 'bold': True}
                    })
                    chart.set_x_axis({
                        'name': 'Test Date',
                        'name_font': {'size': 10},
                        'num_font': {'size': 9, 'rotation': 45}
                    })
                    chart.set_y_axis({
                        'name': f'Value {ref_range}' if ref_range else 'Value',
                        'name_font': {'size': 10},
                        'min': y_min,
                        'max': y_max
                    })
                    
                    # Show legend if we have reference ranges
                    if ref_type in ["range", "less_than", "greater_than"]:
                        chart.set_legend({
                            'position': 'bottom',
                            'font': {'size': 8}
                        })
                    else:
                        chart.set_legend({'none': True})
                        
                    chart.set_size({'width': chart_width, 'height': chart_height})
                    chart.set_style(10)  # Use a clean style
                    
                    # Insert chart
                    chart_worksheet.insert_chart(chart_start_row, chart_start_col, chart)
                    
                    chart_count += 1
        
        # Add summary information to charts sheet
        if chart_count > 0:
            summary_row = 2 + ((chart_count // charts_per_row + 1) * chart_spacing_y)
            chart_worksheet.write(summary_row, 0, f'Total Charts Generated: {chart_count}', 
                                workbook.add_format({'bold': True, 'font_size': 12}))
            chart_worksheet.write(summary_row + 1, 0, 'Note: Only tests with multiple numeric values are charted', 
                                workbook.add_format({'italic': True, 'font_color': '#666666'}))
        else:
            chart_worksheet.write(4, 0, 'No charts could be generated - insufficient numeric data points', 
                                workbook.add_format({'font_color': 'red', 'bold': True}))
        
        # =================== END CHARTS TAB ===================
        
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
        summary_sheet.write('A14', f"Charts Generated: {chart_count}")
        
        if date_lab_cols_sorted:
            first_date = date_lab_cols_sorted[0].split('_')[0]
            last_date = date_lab_cols_sorted[-1].split('_')[0]
            summary_sheet.write('A15', f"Date Range: {first_date} to {last_date}")
        
        summary_sheet.write('A16', f"Generated on: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}")
        
        # Test categories
        summary_sheet.write('A18', 'Test Categories:', info_format)
        categories = organized_df['Test_Category'].value_counts()
        for i, (category, count) in enumerate(categories.items()):
            summary_sheet.write(f'A{19+i}', f"• {category}: {count} tests")
        
        # Labs used
        summary_sheet.write('A' + str(19 + len(categories) + 2), 'Labs Used:', info_format)
        unique_labs = set()
        for col in date_lab_cols_sorted:
            if '_' in col:
                lab_part = col.split('_', 1)[1]
                unique_labs.add(lab_part)
        
        for i, lab in enumerate(sorted(unique_labs)):
            summary_sheet.write(f'A{19 + len(categories) + 3 + i}', f"• {lab}")

    return output_excel.getvalue()

    
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

def validate_data_consistency(df):
    """Validate the consistency of the combined dataset"""
    issues = []
    
    if df.empty:
        return ["No data available"]
    
    # Check for missing critical fields
    critical_fields = ['Test_Name', 'Result', 'Test_Date']
    for field in critical_fields:
        missing_count = df[field].isna().sum() + (df[field] == 'N/A').sum()
        if missing_count > 0:
            issues.append(f"Missing {field} in {missing_count} records")
    
    # Check date consistency
    date_issues = df[df['Test_Date_dt'].isna() & (df['Test_Date'] != 'N/A')]
    if not date_issues.empty:
        issues.append(f"Invalid date formats in {len(date_issues)} records")
    
    # Check for duplicate test entries on same date
    duplicates = df.groupby(['Test_Name', 'Test_Date']).size()
    duplicate_count = (duplicates > 1).sum()
    if duplicate_count > 0:
        issues.append(f"Found {duplicate_count} duplicate test entries for same dates")
    
    return issues

def display_combination_summary(existing_count, new_count, total_count, patient_info):
    """Display a nice summary of the data combination process"""
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="📋 Existing Records", 
            value=existing_count,
            help="Records from uploaded Excel/CSV file"
        )
    
    with col2:
        st.metric(
            label="📄 New PDF Records", 
            value=new_count,
            help="Records extracted from new PDF reports"
        )
    
    with col3:
        st.metric(
            label="📊 Total Records", 
            value=total_count,
            delta=f"+{new_count}" if new_count > 0 else None,
            help="Combined total records available"
        )
    
    # Show patient info summary
    if patient_info:
        st.info(f"👤 **Patient**: {patient_info.get('name', 'N/A')} | "
                f"📅 **Latest Date**: {patient_info.get('date', 'N/A')} | "
                f"🏥 **Primary Lab**: {patient_info.get('lab_name', 'N/A')}")
def safe_file_processing(file_processor_func, file_obj, *args, **kwargs):
    """Safely process files with comprehensive error handling"""
    try:
        return file_processor_func(file_obj, *args, **kwargs)
    except pd.errors.EmptyDataError:
        st.error(f"❌ File {file_obj.name} appears to be empty or corrupted")
        return pd.DataFrame(), {}
    except pd.errors.ParserError as e:
        st.error(f"❌ Could not parse file {file_obj.name}: {str(e)}")
        return pd.DataFrame(), {}
    except Exception as e:
        st.error(f"❌ Unexpected error processing {file_obj.name}: {str(e)}")
        return pd.DataFrame(), {}


def handle_patient_info_conflicts(existing_info, new_info_list):
    """Handle conflicts in patient information with user-friendly messages"""
    
    if not existing_info and not new_info_list:
        return {}
    
    # Collect all names for comparison
    all_names = []
    if existing_info.get('name') and existing_info.get('name') != 'N/A':
        all_names.append(existing_info.get('name'))
    
    for info in new_info_list:
        if info.get('name') and info.get('name') != 'N/A':
            all_names.append(info.get('name'))
    
    # Check for name conflicts
    normalized_names = [normalize_name(name) for name in all_names]
    unique_names = set(normalized_names)
    
    if len(unique_names) > 1:
        st.warning("⚠️ **Patient Name Conflict Detected**")
        
        with st.expander("🔍 View Name Conflict Details", expanded=False):
            st.write("**Names found across all sources:**")
            for i, name in enumerate(set(all_names), 1):
                st.write(f"{i}. {name}")
            
            st.write("**Normalized names:**")
            for i, name in enumerate(unique_names, 1):
                st.write(f"{i}. {name}")
        
        # Use smart consolidation
        consolidated = smart_consolidate_patient_info(existing_info, new_info_list)
        
        st.success(f"✅ **Auto-resolved to**: {consolidated.get('name', 'N/A')}")
        st.info("💡 The system selected the most complete and frequently occurring name")
        
        return consolidated
    
    else:
        # No conflicts, proceed with normal consolidation
        return smart_consolidate_patient_info(existing_info, new_info_list)

def enhanced_data_combination_workflow(uploaded_excel_file, new_pdf_data_list, new_patient_info_list):
    """
    FIXED: Main workflow to combine existing Excel/CSV with new PDF data
    """
    print("=== Starting Data Combination Workflow ===")
    
    # Process existing Excel/CSV file
    print(f"Processing existing file: {uploaded_excel_file.name}")
    existing_df, existing_patient_info = process_existing_excel_csv(uploaded_excel_file, new_patient_info_list)
    
    print(f"Existing data shape: {existing_df.shape}")
    print(f"Existing data columns: {list(existing_df.columns) if not existing_df.empty else 'Empty DataFrame'}")
    
    # Process new PDF data
    combined_df = pd.DataFrame()
    
    if not existing_df.empty:
        print("Adding existing data to combined dataset...")
        combined_df = existing_df.copy()
        print(f"Combined data after adding existing: {combined_df.shape}")
    
    # Add new PDF data if available
    if new_pdf_data_list:
        print(f"Processing {len(new_pdf_data_list)} new PDF datasets...")
        new_data_df = pd.concat(new_pdf_data_list, ignore_index=True)
        print(f"New PDF data shape: {new_data_df.shape}")
        
        if not combined_df.empty:
            print("Combining existing + new data...")
            # Ensure both dataframes have the same columns
            all_columns = list(set(combined_df.columns.tolist() + new_data_df.columns.tolist()))
            
            # Add missing columns to both dataframes
            for col in all_columns:
                if col not in combined_df.columns:
                    combined_df[col] = 'N/A'
                if col not in new_data_df.columns:
                    new_data_df[col] = 'N/A'
            
            # Reorder columns to match
            combined_df = combined_df[all_columns]
            new_data_df = new_data_df[all_columns]
            
            # Combine the data
            combined_df = pd.concat([combined_df, new_data_df], ignore_index=True)
            print(f"Final combined data shape: {combined_df.shape}")
        else:
            print("No existing data, using only new PDF data...")
            combined_df = new_data_df
    else:
        print("No new PDF data to add...")
    
    # Smart consolidation of patient info
    final_patient_info = smart_consolidate_patient_info(existing_patient_info, new_patient_info_list)
    
    print(f"=== Final Results ===")
    print(f"Combined data shape: {combined_df.shape}")
    print(f"Patient info: {final_patient_info}")
    
    return combined_df, final_patient_info

def process_existing_excel_csv(uploaded_excel_file, new_patient_info_list):
    """
    FIXED: Enhanced function to properly process existing Excel/CSV files and convert them
    to the same format as PDF-extracted data
    """
    try:
        filename = uploaded_excel_file.name
        print(f"Processing file: {filename}")
        
        # Read the file
        if filename.endswith('.csv'):
            df = pd.read_csv(uploaded_excel_file)
            print(f"Read CSV file with shape: {df.shape}")
        else:
            # For Excel files, try to read from the main data sheet
            try:
                # Try to read from 'Medical Data with Trends' sheet first (our standard format)
                df = pd.read_excel(uploaded_excel_file, sheet_name='Medical Data with Trends', skiprows=3)
                print("Read from 'Medical Data with Trends' sheet")
            except:
                try:
                    # If that fails, try the first sheet
                    df = pd.read_excel(uploaded_excel_file, sheet_name=0)
                    print("Read from first sheet")
                except:
                    # If that also fails, try without specifying sheet
                    df = pd.read_excel(uploaded_excel_file)
                    print("Read using default method")
        
        print(f"Initial dataframe shape: {df.shape}")
        print(f"Initial columns: {list(df.columns)}")
        
        # Clean up the dataframe
        df = df.dropna(how='all').reset_index(drop=True)
        print(f"After cleaning shape: {df.shape}")
        
        # Check if this is already in normalized format (has columns like Test_Date, Result, etc.)
        required_normalized_cols = ['Test_Name', 'Result', 'Test_Date']
        has_normalized_format = all(col in df.columns for col in required_normalized_cols)
        
        print(f"Has normalized format: {has_normalized_format}")
        
        if has_normalized_format:
            print("File is already in normalized format, processing directly...")
            return process_normalized_excel_data(df, filename, new_patient_info_list)
        
        # Check if this is in pivoted format (Test_Category, Test_Name as first columns, then date columns)
        has_pivoted_format = 'Test_Category' in df.columns and 'Test_Name' in df.columns
        print(f"Has pivoted format: {has_pivoted_format}")
        
        if has_pivoted_format:
            print("File is in pivoted format, converting to normalized format...")
            return process_pivoted_excel_data(df, filename, new_patient_info_list)
        
        # If neither format is detected, try to auto-detect
        print("Auto-detecting file format...")
        return auto_detect_and_process(df, filename, new_patient_info_list)
        
    except Exception as e:
        print(f"Error processing existing file: {str(e)}")
        st.error(f"Error processing existing file: {str(e)}")
        return pd.DataFrame(), {}

def process_normalized_excel_data(df, filename, new_patient_info_list):
    """
    FIXED: Process Excel data that's already in normalized format
    """
    print("Processing normalized Excel data...")
    
    # Ensure all required columns exist
    expected_columns = [
        'Source_Filename', 'Patient_ID', 'Patient_Name', 'Age', 'Gender', 
        'Test_Date', 'Lab_Name', 'Test_Category', 'Original_Test_Name', 
        'Test_Name', 'Result', 'Unit', 'Reference_Range', 'Status', 
        'Processed_Date', 'Result_Numeric', 'Test_Date_dt'
    ]
    
    # Add missing columns with default values
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 'N/A'
            print(f"Added missing column: {col}")
    
    # Update computed columns
    df['Source_Filename'] = filename
    df['Processed_Date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Fix numeric conversion for Result column
    df['Result_Numeric'] = pd.to_numeric(df['Result'], errors='coerce')
    
    # Fix date parsing
    df['Test_Date_dt'] = df['Test_Date'].apply(parse_date_dd_mm_yyyy)
    
    # Clean up the dataframe - remove completely empty rows
    df = df.dropna(subset=['Test_Name'], how='all')
    df = df[df['Test_Name'] != 'N/A']
    df = df[df['Test_Name'].notna()]
    
    print(f"Processed normalized data shape: {df.shape}")
    
    # Extract patient info from the existing data
    existing_patient_info = extract_patient_info_from_normalized_data(df)
    
    # Reorder columns to match expected format
    df = df[expected_columns]
    
    return df, existing_patient_info

def process_pivoted_excel_data(df, filename, new_patient_info_list):
    """
    FIXED: Process Excel data that's in pivoted format (organized by date)
    """
    print("Processing pivoted Excel data...")
    
    # Remove header rows if they exist (Date and Lab rows)
    original_shape = df.shape
    
    # Check for and remove header rows that contain dates or lab info
    rows_to_skip = 0
    for idx in range(min(3, len(df))):  # Check first 3 rows
        row_content = str(df.iloc[idx, 0]).lower() if not pd.isna(df.iloc[idx, 0]) else ""
        if any(indicator in row_content for indicator in ['📅', 'date', '🏥', 'lab']):
            rows_to_skip = idx + 1
    
    if rows_to_skip > 0:
        df = df.iloc[rows_to_skip:].reset_index(drop=True)
        print(f"Skipped {rows_to_skip} header rows. New shape: {df.shape}")
    
    # Identify date-lab columns (everything except Test_Category and Test_Name)
    non_date_cols = ['Test_Category', 'Test_Name']
    date_lab_cols = [col for col in df.columns if col not in non_date_cols and not col.startswith('Unnamed')]
    
    print(f"Found {len(date_lab_cols)} date-lab columns: {date_lab_cols[:5]}...")  # Show first 5
    
    if not date_lab_cols:
        print("No date columns found in pivoted data")
        return pd.DataFrame(), {}
    
    # Convert from pivoted to normalized format
    normalized_rows = []
    
    print(f"Processing {len(df)} test rows...")
    
    for idx, row in df.iterrows():
        if idx % 10 == 0:  # Progress indicator
            print(f"Processing row {idx}/{len(df)}")
            
        test_category = row.get('Test_Category', 'N/A')
        test_name = row.get('Test_Name', 'N/A')
        
        # Skip if test name is not valid
        if pd.isna(test_name) or test_name in ['N/A', '', 'Test_Name'] or str(test_name).strip() == '':
            continue
        
        for date_lab_col in date_lab_cols:
            result_value = row.get(date_lab_col, '')
            
            # Skip empty results
            if pd.isna(result_value) or result_value in ['', 'N/A'] or str(result_value).strip() == '':
                continue
            
            # Parse date and lab from column name
            if '_' in str(date_lab_col):
                parts = str(date_lab_col).split('_', 1)
                test_date = parts[0] if len(parts) > 0 else 'N/A'
                lab_name = parts[1] if len(parts) > 1 else 'N/A'
            else:
                # If no underscore, assume it's just a date
                test_date = str(date_lab_col)
                lab_name = 'N/A'
            
            # Normalize the date format
            parsed_date = parse_date_dd_mm_yyyy(test_date)
            formatted_date = format_date_dd_mm_yyyy(parsed_date) if parsed_date else test_date
            
            # Determine patient info - use from new PDFs if available, otherwise extract from filename
            patient_name = 'N/A'
            patient_age = 'N/A'
            patient_gender = 'N/A'
            patient_id = 'N/A'
            
            if new_patient_info_list:
                # Use the most recent patient info from new PDFs
                latest_info = new_patient_info_list[-1]  # Assuming the last one is most recent
                patient_name = latest_info.get('name', extract_patient_info_from_excel_filename(filename))
                patient_age = latest_info.get('age', 'N/A')
                patient_gender = latest_info.get('gender', 'N/A')
                patient_id = latest_info.get('patient_id', 'N/A')
            else:
                patient_name = extract_patient_info_from_excel_filename(filename)
            
            normalized_row = {
                'Source_Filename': filename,
                'Patient_ID': patient_id,
                'Patient_Name': patient_name,
                'Age': patient_age,
                'Gender': patient_gender,
                'Test_Date': formatted_date,
                'Lab_Name': lab_name,
                'Test_Category': test_category,
                'Original_Test_Name': test_name,
                'Test_Name': test_name,  # Will be standardized later if needed
                'Result': str(result_value),
                'Unit': 'N/A',  # Not available in pivoted format
                'Reference_Range': 'N/A',  # Not available in pivoted format
                'Status': 'N/A',  # Not available in pivoted format
                'Processed_Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'Result_Numeric': pd.to_numeric(result_value, errors='coerce'),
                'Test_Date_dt': parsed_date
            }
            
            normalized_rows.append(normalized_row)
    
    print(f"Created {len(normalized_rows)} normalized rows")
    
    if not normalized_rows:
        print("No valid data found in pivoted format")
        return pd.DataFrame(), {}
    
    normalized_df = pd.DataFrame(normalized_rows)
    
    # Extract patient info (use the most recent date)
    if normalized_df.empty:
        existing_patient_info = {}
    else:
        # Get the most recent date entry for patient info
        most_recent_idx = normalized_df['Test_Date_dt'].idxmax() if normalized_df['Test_Date_dt'].notna().any() else 0
        recent_row = normalized_df.iloc[most_recent_idx] if most_recent_idx is not None else normalized_df.iloc[0]
        
        existing_patient_info = {
            'name': recent_row['Patient_Name'],
            'age': recent_row['Age'],
            'gender': recent_row['Gender'],
            'patient_id': recent_row['Patient_ID'],
            'date': recent_row['Test_Date'],
            'lab_name': recent_row['Lab_Name']
        }
    
    print(f"Final normalized data shape: {normalized_df.shape}")
    return normalized_df, existing_patient_info

def debug_data_combination(existing_df, new_pdf_data_list, step_name=""):
    """
    Helper function to debug data combination issues
    """
    print(f"\n=== DEBUG: {step_name} ===")
    
    if not existing_df.empty:
        print(f"Existing data: {existing_df.shape} rows")
        print(f"Existing columns: {list(existing_df.columns)}")
        print(f"Sample existing data:")
        print(existing_df[['Test_Name', 'Result', 'Test_Date']].head(3).to_string())
    else:
        print("Existing data: EMPTY")
    
    if new_pdf_data_list:
        total_new_rows = sum(len(df) for df in new_pdf_data_list)
        print(f"New PDF data: {len(new_pdf_data_list)} dataframes, {total_new_rows} total rows")
        if new_pdf_data_list:
            sample_df = new_pdf_data_list[0]
            print(f"Sample new data:")
            print(sample_df[['Test_Name', 'Result', 'Test_Date']].head(3).to_string())
    else:
        print("New PDF data: NONE")
    
    print("=" * 50)

# Add this debugging version of the main workflow
def enhanced_data_combination_workflow_debug(uploaded_excel_file, new_pdf_data_list, new_patient_info_list):
    """
    DEBUG VERSION: Main workflow with extensive logging
    """
    print("=== STARTING DEBUG DATA COMBINATION WORKFLOW ===")
    
    # Step 1: Process existing Excel/CSV file
    print(f"\nSTEP 1: Processing existing file: {uploaded_excel_file.name}")
    existing_df, existing_patient_info = process_existing_excel_csv(uploaded_excel_file, new_patient_info_list)
    
    debug_data_combination(existing_df, new_pdf_data_list, "After processing existing file")
    
    # Step 2: Initialize combined dataframe
    combined_df = pd.DataFrame()
    
    # Step 3: Add existing data
    if not existing_df.empty:
        print(f"\nSTEP 3: Adding existing data to combined dataset...")
        combined_df = existing_df.copy()
        print(f"Combined data after adding existing: {combined_df.shape}")
        print(f"Sample combined data:")
        if not combined_df.empty:
            print(combined_df[['Test_Name', 'Result', 'Test_Date']].head(3).to_string())
    else:
        print(f"\nSTEP 3: No existing data to add (existing_df is empty)")
    
    # Step 4: Add new PDF data if available
    if new_pdf_data_list:
        print(f"\nSTEP 4: Processing {len(new_pdf_data_list)} new PDF datasets...")
        new_data_df = pd.concat(new_pdf_data_list, ignore_index=True)
        print(f"New PDF data shape: {new_data_df.shape}")
        
        if not combined_df.empty:
            print("STEP 4a: Combining existing + new data...")
            
            # Show column comparison
            existing_cols = set(combined_df.columns)
            new_cols = set(new_data_df.columns)
            print(f"Existing columns: {len(existing_cols)}")
            print(f"New columns: {len(new_cols)}")
            print(f"Common columns: {len(existing_cols & new_cols)}")
            print(f"Missing from existing: {new_cols - existing_cols}")
            print(f"Missing from new: {existing_cols - new_cols}")
            
            # Ensure both dataframes have the same columns
            all_columns = list(existing_cols | new_cols)
            
            # Add missing columns to both dataframes
            for col in all_columns:
                if col not in combined_df.columns:
                    combined_df[col] = 'N/A'
                    print(f"Added column '{col}' to existing data")
                if col not in new_data_df.columns:
                    new_data_df[col] = 'N/A'
                    print(f"Added column '{col}' to new data")
            
            # Reorder columns to match
            combined_df = combined_df[all_columns]
            new_data_df = new_data_df[all_columns]
            
            print(f"Before concat - Combined: {combined_df.shape}, New: {new_data_df.shape}")
            
            # Combine the data
            combined_df = pd.concat([combined_df, new_data_df], ignore_index=True)
            print(f"After concat - Final combined data shape: {combined_df.shape}")
            
        else:
            print("STEP 4b: No existing data, using only new PDF data...")
            combined_df = new_data_df
    else:
        print(f"\nSTEP 4: No new PDF data to add...")
    
    # Step 5: Final validation
    print(f"\nSTEP 5: Final validation")
    print(f"Final combined data shape: {combined_df.shape}")
    
    if not combined_df.empty:
        print("Sample final data:")
        print(combined_df[['Source_Filename', 'Test_Name', 'Result', 'Test_Date']].head(5).to_string())
        
        # Check data sources
        source_counts = combined_df['Source_Filename'].value_counts()
        print(f"\nData sources breakdown:")
        for source, count in source_counts.items():
            print(f"  {source}: {count} records")
    else:
        print("WARNING: Final combined data is EMPTY!")
    
    # Smart consolidation of patient info
    final_patient_info = smart_consolidate_patient_info(existing_patient_info, new_patient_info_list)
    
    print(f"\n=== FINAL RESULTS ===")
    print(f"Combined data shape: {combined_df.shape}")
    print(f"Patient info: {final_patient_info}")
    print("=== END DEBUG WORKFLOW ===\n")
    
    return combined_df, final_patient_info

# Replace the existing parse_date_dd_mm_yyyy function if it has issues
def parse_date_dd_mm_yyyy(date_str):
    """FIXED: Parse date string ensuring DD/MM/YYYY format"""
    if not date_str or pd.isna(date_str) or date_str in ['N/A', '']:
        return None
    
    # Convert to string in case it's not already
    date_str = str(date_str).strip()
    
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
                    print(f"Could not parse date: {date_str}")
                    return None

