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

# Add the script's directory to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)



# Initialize Gemini models globally
gemini_model_extraction = None
gemini_model_chat = None



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

# Replace the existing analysis workflow in your streamlit app with this enhanced version:

if st.button("🔬 Analyze Reports", key="analyze_btn"):
    st.session_state.analysis_done = False
    st.session_state.report_df = pd.DataFrame()
    st.session_state.consolidated_patient_info = {}
    st.session_state.chat_history = []
    st.session_state.raw_texts = []

    all_dfs = []
    all_patient_infos_from_pdfs = []

    with st.spinner("Processing reports and analyzing with AI... This may take a moment."):
        # First, process new PDF files if any
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

        # Process existing Excel/CSV file if provided
        existing_df = pd.DataFrame()
        existing_patient_info = {}
        
        if upload_mode == "Add new medical reports to an existing Excel/CSV file" and uploaded_excel_file_object:
            try:
                st.write(f"--- Processing existing file: {uploaded_excel_file_object.name} ---")
                
                # Use the enhanced processing function
                existing_df, existing_patient_info = process_existing_excel_csv(
                    uploaded_excel_file_object, 
                    all_patient_infos_from_pdfs
                )
                
                if not existing_df.empty:
                    st.success(f"✅ Successfully processed existing file: {uploaded_excel_file_object.name}")
                    st.info(f"📊 Found {len(existing_df)} existing test records")
                else:
                    st.warning(f"⚠️ No data could be extracted from the existing file: {uploaded_excel_file_object.name}")

            except Exception as e:
                st.error(f"❌ Error processing existing Excel/CSV file: {str(e)}")
                st.info("Continuing with only new PDF data...")

        # Combine all data
        combined_raw_df = pd.DataFrame()
        
        if not existing_df.empty and all_dfs:
            # Both existing and new data
            new_data_df = pd.concat(all_dfs, ignore_index=True)
            combined_raw_df = pd.concat([existing_df, new_data_df], ignore_index=True)
            st.info(f"📊 Combined {len(existing_df)} existing records with {len(new_data_df)} new records")
            
        elif not existing_df.empty:
            # Only existing data
            combined_raw_df = existing_df
            st.info(f"📊 Using {len(existing_df)} existing records")
            
        elif all_dfs:
            # Only new data
            combined_raw_df = pd.concat(all_dfs, ignore_index=True)
            st.info(f"📊 Processing {len(combined_raw_df)} new records")
            
        else:
            # No data at all
            st.warning("⚠️ No data found in either existing files or new PDFs")

        # Smart consolidation of patient information
        if existing_patient_info and all_patient_infos_from_pdfs:
            # Both sources available - use smart consolidation
            st.info("🔄 Consolidating patient information from existing data and new PDFs...")
            consolidated_info = smart_consolidate_patient_info(existing_patient_info, all_patient_infos_from_pdfs)
            
            # Check for potential name conflicts and resolve
            all_patient_sources = [existing_patient_info] + all_patient_infos_from_pdfs
            all_names = [pi.get('name', '') for pi in all_patient_sources if pi.get('name') and pi.get('name') != 'N/A']
            
            if len(set([normalize_name(name) for name in all_names])) > 1:
                st.warning("⚠️ Multiple patient names detected across existing and new data!")
                st.write("Names found:", ", ".join(set(all_names)))
                st.info("🔄 Using smart name resolution to select the most complete name...")
                
        elif existing_patient_info:
            # Only existing patient info
            consolidated_info = existing_patient_info
            st.info("📋 Using patient information from existing data")
            
        elif all_patient_infos_from_pdfs:
            # Only new patient info from PDFs
            consolidated_info = consolidate_patient_info(all_patient_infos_from_pdfs)
            if 'error' in consolidated_info and consolidated_info['error'] == 'name_mismatch':
                st.warning("⚠️ Different patient names detected in new PDF reports!")
                st.write("Found these different names:", ", ".join(consolidated_info['names']))
                st.info("🔄 Automatically resolving name conflicts using smart selection...")
                consolidated_info = create_consolidated_info_with_smart_selection(all_patient_infos_from_pdfs)
                st.success(f"✅ Resolved name conflict. Using: **{consolidated_info.get('name', 'N/A')}**")
        else:
            # No patient info available
            consolidated_info = {}
            st.warning("⚠️ No patient information could be extracted")

        # Final data validation and cleanup
        if not combined_raw_df.empty:
            # Remove any completely empty rows
            combined_raw_df = combined_raw_df.dropna(subset=['Test_Name', 'Result'], how='all')
            
            # Ensure consistent data types
            combined_raw_df['Result_Numeric'] = pd.to_numeric(combined_raw_df['Result'], errors='coerce')
            combined_raw_df['Test_Date_dt'] = combined_raw_df['Test_Date'].apply(parse_date_dd_mm_yyyy)
            
            # Sort by date and category for consistency
            combined_raw_df = combined_raw_df.sort_values(
                by=['Test_Date_dt', 'Test_Category', 'Test_Name'], 
                na_position='last'
            ).reset_index(drop=True)
            
            # Update session state
            st.session_state.report_df = combined_raw_df
            st.session_state.consolidated_patient_info = consolidated_info
            st.session_state.analysis_done = True
            
            # Success message
            st.balloons()
            st.success(f"🎉 Analysis Complete! Processed {len(combined_raw_df)} total test records")
            
            if existing_patient_info and all_patient_infos_from_pdfs:
                st.info(f"📅 Data spans from existing records to new reports dated {consolidated_info.get('date', 'N/A')}")
                
        else:
            st.error("❌ No valid data could be extracted or combined from the provided files")
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