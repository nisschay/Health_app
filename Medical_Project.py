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
from Helper_Functions import *
import sys
import os
from collections import Counter
import hashlib
import concurrent.futures
from functools import lru_cache

try:
    from backend_api.app.normalization import normalize_dataframe
except Exception:
    def normalize_dataframe(df, *, deduplicate=True):
        return df

# Add the script's directory to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

# Initialize Gemini models globally
gemini_model_extraction = None
gemini_model_chat = None

# --- Performance: Caching functions ---
@st.cache_data(ttl=3600, show_spinner=False)
def cached_extract_text_from_pdf(file_content_hash, file_content):
    """Cache PDF text extraction to avoid re-processing same files"""
    return extract_text_from_pdf(file_content)

@st.cache_data(ttl=3600, show_spinner=False)
def cached_analyze_report(text_hash, report_text, api_key):
    """Cache AI analysis results for same content"""
    return analyze_medical_report_with_gemini(report_text, api_key)

def get_file_hash(file_content):
    """Generate hash for file content to use as cache key"""
    return hashlib.md5(file_content).hexdigest()

# --- Streamlit App UI ---
st.set_page_config(
    page_title="Medical Report Analyzer",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load custom CSS for modern UI
def load_css():
    css_path = os.path.join(script_dir, 'style.css')
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css()

# --- Hero Section ---
st.markdown("""
<div style="text-align: center; padding: 1rem 0 2rem 0;">
    <h1 style="margin-bottom: 0.5rem;">🏥 Medical Report Analyzer</h1>
    <p style="font-size: 1.1rem; color: #94a3b8; max-width: 600px; margin: 0 auto;">
        Upload your medical PDF reports to get structured data, AI-powered health insights, and beautiful visualizations.
    </p>
</div>
""", unsafe_allow_html=True)

# --- Gemini API Key ---
api_key = st.secrets.get("GEMINI_API_KEY")
if not api_key or not init_gemini_models(api_key):
    st.error("🚨 Unable to initialize AI models. Please check the server configuration.")
    st.stop()

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
if 'processing_progress' not in st.session_state:
    st.session_state.processing_progress = 0

# --- UI for Upload ---
# Center the radio buttons to match the rest of the layout
col_spacer1, col_radio, col_spacer2 = st.columns([1, 2, 1])

with col_radio:
    upload_mode = st.radio(
        "What would you like to do?",
        ["Upload new medical reports", "Add reports to existing data file"],
        index=0,
        key="upload_mode_selector",
        horizontal=True
    )

uploaded_excel_file_object = None
if upload_mode == "Add reports to existing data file":
    # Center the Excel uploader
    col_s1, col_excel, col_s2 = st.columns([1, 2, 1])
    with col_excel:
        uploaded_excel_file_object = st.file_uploader(
            "Upload your previously downloaded Excel or CSV file",
            type=["csv", "xlsx"],
            key="excel_uploader",
            help="Upload the Excel/CSV file you downloaded from a previous analysis session"
        )

# Center the PDF uploader
col_s1, col_pdf, col_s2 = st.columns([1, 2, 1])
with col_pdf:
    uploaded_files = st.file_uploader(
        "Upload Medical Report PDFs",
        type="pdf",
        accept_multiple_files=True,
        key="pdf_uploader",
        help="You can upload multiple PDF files at once. We support most medical lab report formats."
    )

# --- Analysis Button - centered, same style as Download button ---
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    analyze_button = st.button(
        "🔬 Analyze Reports",
        key="analyze_btn",
        use_container_width=True,
        type="primary"
    )

if analyze_button:
    if not uploaded_files and not uploaded_excel_file_object:
        st.warning("⚠️ Please upload at least one PDF report or an existing data file to analyze.")
        st.stop()
    
    # Reset session state
    st.session_state.analysis_done = False
    st.session_state.report_df = pd.DataFrame()
    st.session_state.consolidated_patient_info = {}
    st.session_state.chat_history = []
    st.session_state.raw_texts = []

    all_dfs = []
    all_patient_infos_from_pdfs = []
    
    # Create progress container
    progress_container = st.container()
    
    with progress_container:
        # Progress bar and status
        progress_bar = st.progress(0, text="Initializing...")
        status_text = st.empty()
        
        total_files = len(uploaded_files) if uploaded_files else 0
        has_excel = uploaded_excel_file_object is not None
        total_steps = total_files + (2 if has_excel else 0) + 2  # +2 for processing steps
        current_step = 0
        
        # Process new PDF files with progress tracking
        if uploaded_files:
            for i, uploaded_file in enumerate(uploaded_files):
                current_step += 1
                progress = current_step / total_steps
                progress_bar.progress(progress, text=f"📄 Processing: {uploaded_file.name}")
                status_text.info(f"Extracting text from {uploaded_file.name}...")
                
                file_content = uploaded_file.read()
                file_hash = get_file_hash(file_content)
                
                # Use cached extraction if available
                report_text = cached_extract_text_from_pdf(file_hash, file_content)
                st.session_state.raw_texts.append({
                    "name": uploaded_file.name,
                    "text": report_text[:2000] if report_text else "No text extracted"
                })
                
                if report_text:
                    status_text.info(f"🤖 AI analyzing {uploaded_file.name}...")
                    text_hash = hashlib.md5(report_text.encode()).hexdigest()
                    gemini_analysis_json = cached_analyze_report(text_hash, report_text, api_key)
                    
                    if gemini_analysis_json:
                        df_single, patient_info_single = create_structured_dataframe(
                            gemini_analysis_json, uploaded_file.name
                        )
                        if not df_single.empty:
                            all_dfs.append(df_single)
                        if patient_info_single:
                            all_patient_infos_from_pdfs.append(patient_info_single)
                        status_text.success(f"✅ {uploaded_file.name} analyzed successfully")
                    else:
                        status_text.error(f"⚠️ Could not analyze {uploaded_file.name}")
                else:
                    status_text.error(f"⚠️ Could not extract text from {uploaded_file.name}")

        # Process existing Excel/CSV file
        existing_df = pd.DataFrame()
        existing_patient_info = {}
        
        if upload_mode == "Add reports to existing data file" and uploaded_excel_file_object:
            current_step += 1
            progress_bar.progress(current_step / total_steps, text="📊 Processing existing data...")
            status_text.info(f"Loading existing file: {uploaded_excel_file_object.name}")
            
            try:
                existing_df, existing_patient_info = process_existing_excel_csv(
                    uploaded_excel_file_object,
                    all_patient_infos_from_pdfs
                )
                
                if not existing_df.empty:
                    status_text.success(f"✅ Loaded {len(existing_df)} existing records")
                else:
                    status_text.warning("⚠️ No data found in existing file")
                    
            except Exception as e:
                status_text.error(f"❌ Error loading existing file: {str(e)}")

        # Combine all data
        current_step += 1
        progress_bar.progress(current_step / total_steps, text="🔄 Combining data...")
        
        combined_raw_df = pd.DataFrame()
        
        if not existing_df.empty and all_dfs:
            new_data_df = pd.concat(all_dfs, ignore_index=True)
            combined_raw_df = pd.concat([existing_df, new_data_df], ignore_index=True)
            status_text.info(f"📊 Combined {len(existing_df)} existing + {len(new_data_df)} new records")
        elif not existing_df.empty:
            combined_raw_df = existing_df
        elif all_dfs:
            combined_raw_df = pd.concat(all_dfs, ignore_index=True)
        else:
            st.error("❌ No data could be extracted from the provided files")
            st.stop()

        # Consolidate patient information
        if existing_patient_info and all_patient_infos_from_pdfs:
            consolidated_info = smart_consolidate_patient_info(existing_patient_info, all_patient_infos_from_pdfs)
        elif existing_patient_info:
            consolidated_info = existing_patient_info
        elif all_patient_infos_from_pdfs:
            consolidated_info = consolidate_patient_info(all_patient_infos_from_pdfs)
            if 'error' in consolidated_info and consolidated_info['error'] == 'name_mismatch':
                consolidated_info = create_consolidated_info_with_smart_selection(all_patient_infos_from_pdfs)
        else:
            consolidated_info = {}

        # Final processing
        current_step += 1
        progress_bar.progress(current_step / total_steps, text="✨ Finalizing analysis...")
        
        if not combined_raw_df.empty:
            # Clean and validate data
            combined_raw_df = combined_raw_df.dropna(subset=['Test_Name', 'Result'], how='all')
            combined_raw_df['Result_Numeric'] = pd.to_numeric(combined_raw_df['Result'], errors='coerce')
            combined_raw_df['Test_Date_dt'] = combined_raw_df['Test_Date'].apply(parse_date_dd_mm_yyyy)
            
            # Sort by date and category
            combined_raw_df = combined_raw_df.sort_values(
                by=['Test_Date_dt', 'Test_Category', 'Test_Name'],
                na_position='last'
            ).reset_index(drop=True)

            combined_raw_df = normalize_dataframe(combined_raw_df)
            
            # Update session state
            st.session_state.report_df = combined_raw_df
            st.session_state.consolidated_patient_info = consolidated_info
            st.session_state.analysis_done = True
            
            # Complete!
            progress_bar.progress(1.0, text="✅ Analysis complete!")
            status_text.empty()
            st.balloons()
            st.success(f"🎉 Successfully analyzed {len(combined_raw_df)} test records!")
        else:
            st.error("❌ No valid data could be extracted")
            st.session_state.analysis_done = False

# --- Display Results Section ---
if st.session_state.analysis_done and not st.session_state.report_df.empty:
    st.session_state.report_df = normalize_dataframe(st.session_state.report_df)
    
    st.divider()
    
    # Patient Information Card
    st.markdown("## 👤 Patient Profile")
    p_info = st.session_state.consolidated_patient_info
    
    if p_info:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("👤 Name", p_info.get('name', "N/A"))
        with col2:
            st.metric("🎂 Age", p_info.get('age', "N/A"))
        with col3:
            st.metric("⚧ Gender", p_info.get('gender', "N/A"))
        with col4:
            st.metric("🏥 Lab", p_info.get('lab_name', "N/A")[:20] + "..." if len(p_info.get('lab_name', '')) > 20 else p_info.get('lab_name', "N/A"))
    else:
        st.info("No patient information available.")
    
    st.divider()

    # --- Chat Assistant Section ---
    st.markdown("## 💬 AI Health Assistant")
    st.markdown("<p style='color: #94a3b8;'>Ask questions about your medical report and get AI-powered insights.</p>", unsafe_allow_html=True)
    
    user_query = st.chat_input("Ask me anything about your health report...")

    if not st.session_state.chat_history:
        st.markdown("##### 💡 Quick Questions")
        example_questions = [
            "📊 Show a general overview of my report",
            "🔬 Explain my blood test results",
            "⚠️ Are there any concerning findings?",
            "📈 Show my test trends over time"
        ]
        cols = st.columns(2)
        for i, question in enumerate(example_questions):
            with cols[i % 2]:
                if st.button(question, key=f"example_q_{i}", use_container_width=True):
                    user_query = question.split(" ", 1)[1]  # Remove emoji

    if user_query:
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.spinner("🤔 Analyzing your report..."):
            assistant_response = get_chatbot_response(
                st.session_state.report_df,
                user_query,
                st.session_state.chat_history[:-1],
                api_key
            )
            st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})

    # Display chat history
    for i in range(0, len(st.session_state.chat_history), 2):
        if i + 1 < len(st.session_state.chat_history):
            with st.chat_message("user"):
                st.write(st.session_state.chat_history[i]["content"])
            with st.chat_message("assistant", avatar="🤖"):
                st.write(st.session_state.chat_history[i + 1]["content"])
    
    st.divider()

    # --- Visualization Section ---
    st.markdown("## 📈 Test Result Visualizations")
    
    viz_container = st.container()
    with viz_container:
        col1, col2 = st.columns([1, 3])
        
        with col1:
            df_for_viz = st.session_state.report_df.copy()
            df_for_viz = df_for_viz[
                ~df_for_viz['Test_Name'].isin(['N/A', 'UnknownTest', 'Unknown Test']) & 
                df_for_viz['Test_Name'].notna()
            ]
            df_for_viz['Test_Category'] = df_for_viz['Test_Category'].fillna('Other').astype(str)

            if not df_for_viz.empty:
                # Body system filter
                all_body_parts = set()
                for category in df_for_viz['Test_Category'].unique():
                    all_body_parts.update(TEST_CATEGORY_TO_BODY_PARTS.get(category, ["General"]))
                
                body_part_options = ["🔍 All Systems"] + [
                    f"{BODY_PARTS_TO_EMOJI.get(bp, '📊')} {bp}" for bp in sorted(all_body_parts)
                ]
                selected_body_part_display = st.selectbox(
                    "Filter by Body System:",
                    options=body_part_options,
                    key="body_part_selector"
                )
                selected_body_part = selected_body_part_display.split(" ", 1)[1] if "All Systems" not in selected_body_part_display else None

                if selected_body_part and selected_body_part != "General":
                    relevant_categories = [
                        cat for cat in df_for_viz['Test_Category'].unique()
                        if selected_body_part in TEST_CATEGORY_TO_BODY_PARTS.get(cat, ["General"])
                    ]
                    df_for_viz = df_for_viz[df_for_viz['Test_Category'].isin(relevant_categories)]

                # Category filter
                available_categories = sorted(df_for_viz['Test_Category'].unique().tolist())
                if available_categories:
                    category_options = ["📋 All Categories"] + available_categories
                    selected_category = st.selectbox(
                        "Select Category:",
                        options=category_options,
                        key="category_selector"
                    )
                    selected_category = None if "All Categories" in selected_category else selected_category

                    # Test selector
                    if selected_category:
                        tests_in_category = sorted(
                            df_for_viz[df_for_viz['Test_Category'] == selected_category]['Test_Name'].unique().tolist()
                        )
                    else:
                        tests_in_category = sorted(df_for_viz['Test_Name'].unique().tolist())
                    
                    if tests_in_category:
                        selected_test = st.selectbox(
                            "Select Test:",
                            options=["-- Select a test --"] + tests_in_category,
                            key="test_selector_viz"
                        )
                        
                        if selected_test and selected_test != "-- Select a test --":
                            test_specific_data = df_for_viz[df_for_viz['Test_Name'] == selected_test]
                            available_dates = sorted(test_specific_data['Test_Date'].unique().tolist(), reverse=True)
                            
                            if len(available_dates) > 1:
                                selected_plot_date = st.selectbox(
                                    "Select Date:",
                                    options=["All Dates (Trend)"] + available_dates,
                                    key="date_selector_viz"
                                )
                                selected_plot_date = "All Dates" if "All Dates" in selected_plot_date else selected_plot_date
                            else:
                                selected_plot_date = available_dates[0] if available_dates else "All Dates"

        with col2:
            if 'selected_test' in locals() and selected_test and selected_test != "-- Select a test --":
                plot = generate_test_plot(df_for_viz, selected_test, selected_plot_date)
                if plot:
                    st.plotly_chart(plot, use_container_width=True, config={'displayModeBar': True})
                else:
                    st.info(f"📊 No plottable data for {selected_test}. Results may be non-numeric or missing reference ranges.")
            else:
                st.markdown("""
                    <div style='text-align: center; padding: 80px 20px; background: #1e293b; border-radius: 16px; border: 1px dashed #334155;'>
                        <div style='font-size: 3rem; margin-bottom: 1rem;'>📊</div>
                        <h3 style='color: #f1f5f9; margin-bottom: 0.5rem;'>Select a Test to Visualize</h3>
                        <p style='color: #94a3b8;'>Choose a test from the dropdown on the left to see charts and trends.</p>
                    </div>
                """, unsafe_allow_html=True)

    st.divider()

    # --- Health Insights Dashboard Section ---
    st.markdown("## Health Insights Dashboard")
    st.markdown("<p style='color: #a3a3a3;'>Overview of your health organized by body system. Only systems with ≥50% abnormal tests are flagged.</p>", unsafe_allow_html=True)
    
    # Generate the health insights dashboard
    try:
        generate_health_insights_dashboard(st.session_state.report_df)
    except Exception as e:
        st.warning(f"Could not generate health insights dashboard: {str(e)}")
    
    st.divider()
    
    # --- PDF Report Download Section ---
    # FIXED UX: Single download button, no two-step process
    # Center-aligned, clean, reliable download action
    st.markdown("## Download Health Report")
    st.markdown("<p style='color: #a3a3a3;'>Download a comprehensive health report with your test results and AI-powered insights.</p>", unsafe_allow_html=True)
    
    # Generate report on-demand with single download button
    # This avoids the confusing two-button flow
    @st.cache_data(show_spinner=False)
    def get_cached_pdf_report(_df_hash, df, patient_info, api_key):
        """Cache the PDF generation to avoid regenerating on each interaction"""
        return generate_pdf_health_report(df, patient_info, api_key)
    
    # Center the download section
    col_spacer1, col_download, col_spacer2 = st.columns([1, 2, 1])
    
    with col_download:
        # Generate report bytes (cached)
        try:
            df_hash = hash(st.session_state.report_df.to_json())
            report_bytes = get_cached_pdf_report(
                df_hash,
                st.session_state.report_df,
                st.session_state.consolidated_patient_info,
                api_key
            )
            
            if report_bytes:
                patient_name_for_file = "".join(
                    c if c.isalnum() else "_" 
                    for c in st.session_state.consolidated_patient_info.get('name', 'health_report')
                )
                
                filename = f"health_report_{patient_name_for_file}.pdf"
                
                # SINGLE download button - generates valid PDF with reportlab
                st.download_button(
                    label="📥 Download Health Report (PDF)",
                    data=report_bytes,
                    file_name=filename,
                    mime='application/pdf',
                    use_container_width=True,
                    type="primary"
                )
            else:
                st.info("Report generation unavailable. Please try again later.")
        except Exception as e:
            st.error(f"Could not prepare report: {str(e)}")
    
    st.divider()

    # --- Organized Data Table Section ---
    st.markdown("## Organized Data by Date")
    
    # --- Organized Data Table ---
    if not st.session_state.report_df.empty:
        try:
            df_with_date_lab = st.session_state.report_df.copy()
            
            # Clean lab name function
            def clean_lab_name(lab_name):
                if pd.isna(lab_name) or lab_name == 'N/A':
                    return 'Unknown Lab'
                lab_name = str(lab_name).strip()
                billing_terms = ['bill', 'billing', 'invoice', 'receipt', 'payment', 'charges', 'collection center', 'collection centre']
                parts = []
                for separator in ['-', '|', ',', '(', ')']:
                    if separator in lab_name:
                        parts = lab_name.split(separator)
                        break
                if not parts:
                    parts = [lab_name]
                cleaned_parts = [
                    part.strip() for part in parts
                    if not any(term in part.lower() for term in billing_terms)
                    and len(part.strip()) > 2
                    and not part.strip().isdigit()
                ]
                return max(cleaned_parts, key=len) if cleaned_parts else lab_name
            
            # Prepare data
            df_with_date_lab['Lab_Name_Clean'] = df_with_date_lab['Lab_Name'].apply(clean_lab_name)
            df_with_date_lab['Test_Date'] = df_with_date_lab['Test_Date'].fillna('Unknown Date')
            df_with_date_lab['Date_Lab'] = df_with_date_lab['Test_Date'].astype(str) + '_' + df_with_date_lab['Lab_Name_Clean'].astype(str)
            
            # Check if we have valid data
            valid_data = df_with_date_lab[
                df_with_date_lab['Test_Name'].notna() & 
                (df_with_date_lab['Test_Name'] != 'N/A') &
                df_with_date_lab['Result'].notna()
            ]
            
            if valid_data.empty:
                st.warning("⚠️ No valid test data available to organize.")
            else:
                # Create pivot tables with proper aggregation
                organized_df = valid_data.pivot_table(
                    index=['Test_Category', 'Test_Name'],
                    columns='Date_Lab',
                    values='Result',
                    aggfunc='first'
                ).reset_index()
                
                ref_range_df = valid_data.pivot_table(
                    index=['Test_Category', 'Test_Name'],
                    columns='Date_Lab',
                    values='Reference_Range',
                    aggfunc='first'
                ).reset_index()
                
                # Get date-lab columns
                date_lab_cols = [col for col in organized_df.columns if col not in ['Test_Category', 'Test_Name']]
                
                # Sort dates safely
                def safe_date_sort(col_name):
                    try:
                        date_part = col_name.split('_')[0]
                        parsed = parse_date_dd_mm_yyyy(date_part)
                        return parsed if parsed else pd.Timestamp.min
                    except:
                        return pd.Timestamp.min
                
                date_lab_cols_sorted = sorted(date_lab_cols, key=safe_date_sort)
                
                # Reorder columns
                required_cols = ['Test_Category', 'Test_Name'] + date_lab_cols_sorted
                organized_df = organized_df.reindex(columns=required_cols, fill_value='')
                ref_range_df = ref_range_df.reindex(columns=required_cols, fill_value='')
                
                # Sort by category and test name
                organized_df = organized_df.sort_values(['Test_Category', 'Test_Name']).reset_index(drop=True)
                ref_range_df = ref_range_df.sort_values(['Test_Category', 'Test_Name']).reset_index(drop=True)
                
                # Validate data structure
                if organized_df.empty:
                    st.warning("⚠️ Could not organize data. Please ensure reports contain valid test results.")
                else:
                    # Create display dataframe with headers
                    display_df = organized_df.copy()
                    date_row = {'Test_Category': '📅 Date', 'Test_Name': ''}
                    lab_row = {'Test_Category': '🏥 Lab', 'Test_Name': ''}
                    
                    for date_lab_col in date_lab_cols_sorted:
                        parts = str(date_lab_col).split('_', 1)
                        date_row[date_lab_col] = parts[0] if len(parts) > 0 else 'N/A'
                        lab_row[date_lab_col] = parts[1] if len(parts) > 1 else 'N/A'
                    
                    header_rows_df = pd.DataFrame([date_row, lab_row])
                    display_df = pd.concat([header_rows_df, display_df], ignore_index=True)
                    
                    st.markdown("<p style='color: #94a3b8;'>Your medical test results organized by test category, test name, and date.</p>", unsafe_allow_html=True)
                    
                    # Style the dataframe
                    def highlight_header_rows(row):
                        if row.name == 0:
                            return ['background-color: #1e3a5f; font-weight: bold; color: #38bdf8'] * len(row)
                        elif row.name == 1:
                            return ['background-color: #1e3a3a; font-weight: bold; color: #10b981; font-style: italic'] * len(row)
                        return [''] * len(row)
                    
                    styled_df = display_df.style.apply(highlight_header_rows, axis=1)
                    st.dataframe(styled_df, use_container_width=True, height=400)

                    # Download buttons
                    patient_name_for_file = "".join(c if c.isalnum() else "_" for c in p_info.get('name', 'medical_data'))
                    
                    try:
                        excel_data = create_enhanced_excel_with_trends(organized_df, ref_range_df, date_lab_cols_sorted, p_info)
                        organized_csv = display_df.to_csv(index=False).encode('utf-8')
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.download_button(
                                label="📥 Download Excel with Charts",
                                data=excel_data,
                                file_name=f"medical_reports_{patient_name_for_file}.xlsx",
                                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                                use_container_width=True
                            )
                        with col2:
                            st.download_button(
                                label="📥 Download CSV",
                                data=organized_csv,
                                file_name=f"medical_data_{patient_name_for_file}.csv",
                                mime='text/csv',
                                use_container_width=True
                            )
                        
                        with st.expander("📊 Excel File Features", expanded=False):
                            st.markdown("""
                            - ✅ Organized data by category & test
                            - ✅ Clean lab facility names  
                            - ✅ Reference ranges in trend charts
                            - ✅ Embedded trend line charts
                            - ✅ Summary sheet with patient info
                            - ✅ Auto-filtering and frozen panes
                            """)
                    except Exception as e:
                        st.error(f"Could not generate download files: {str(e)}")
                
        except Exception as e:
            st.error(f"Error organizing data: {str(e)}")
            st.info("💡 Try re-uploading your reports. Some report formats may need additional processing.")
    else:
        st.info("📤 Upload and analyze reports to see organized data here.")

    # --- Raw Data Section ---
    st.divider()
    st.markdown("## 🗂️ Detailed Report Data")
    
    with st.expander("📋 View Raw Extracted Data", expanded=False):
        display_cols = [col for col in st.session_state.report_df.columns if col not in ['Result_Numeric', 'Test_Date_dt']]
        st.dataframe(st.session_state.report_df[display_cols], use_container_width=True)
        
        csv_export = st.session_state.report_df.to_csv(index=False).encode('utf-8')
        patient_name_for_file = "".join(c if c.isalnum() else "_" for c in p_info.get('name', 'medical_data'))
        st.download_button(
            label="📥 Download Complete Data as CSV",
            data=csv_export,
            file_name=f"complete_medical_data_{patient_name_for_file}.csv",
            mime='text/csv',
        )
    
    with st.expander("📄 View Raw PDF Text Extracts", expanded=False):
        if st.session_state.raw_texts:
            for item in st.session_state.raw_texts:
                st.markdown(f"**{item['name']}**")
                st.text_area("", item['text'], height=150, disabled=True, key=f"raw_text_{item['name']}")
        else:
            st.info("No raw text snippets available.")

elif st.session_state.analysis_done and st.session_state.report_df.empty:
    st.warning("⚠️ Analysis completed but no data was extracted. Please check your PDF files and try again.")
else:
    # Welcome state - show helpful info
    st.markdown("""
    <div style='background: #1e293b; border-radius: 16px; padding: 2rem; margin-top: 2rem; border: 1px solid #334155;'>
        <h3 style='color: #f1f5f9; margin-bottom: 1rem;'>👋 Getting Started</h3>
        <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1.5rem;'>
            <div>
                <h4 style='color: #38bdf8;'>📤 Step 1: Upload</h4>
                <p style='color: #94a3b8;'>Upload your medical report PDFs above. We support most lab report formats.</p>
            </div>
            <div>
                <h4 style='color: #10b981;'>🔬 Step 2: Analyze</h4>
                <p style='color: #94a3b8;'>Click the analyze button to extract and process your health data with AI.</p>
            </div>
            <div>
                <h4 style='color: #8b5cf6;'>📊 Step 3: Explore</h4>
                <p style='color: #94a3b8;'>View charts, trends, and ask questions about your health data.</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)