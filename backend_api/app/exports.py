"""The PDF and Excel files a user downloads."""
from __future__ import annotations

import io
from collections import Counter
from datetime import datetime

import pandas as pd


def create_enhanced_excel_with_trends(organized_df, ref_range_df, date_lab_cols_sorted, patient_info):
    from xlsxwriter.utility import xl_col_to_name

    output_excel = io.BytesIO()
    with pd.ExcelWriter(output_excel, engine='xlsxwriter') as writer:
        # Write the data without headers first, starting from row 4 (index 3)
        organized_df.to_excel(writer, index=False, sheet_name='Medical Data with Trends', startrow=3, header=False)

        workbook = writer.book
        worksheet = writer.sheets['Medical Data with Trends']

        # Title format
        title_format = workbook.add_format({'bold': True, 'font_size': 16, 'align': 'center', 'bg_color': '#4472C4', 'font_color': 'white'})
        last_column_letter = xl_col_to_name(max(len(organized_df.columns) - 1, 0))
        worksheet.merge_range(f'A1:{last_column_letter}1', 'Medical Test Results - Organized by Date', title_format)

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
        for i, _date_lab_col in enumerate(date_lab_cols_sorted):
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
                for col_idx, _col_name in enumerate(date_lab_cols_sorted):
                    if col_idx + 2 < len(ref_range_df.columns):
                        ref_value = ref_range_df.iloc[row_num, col_idx + 2]
                        if pd.notna(ref_value) and str(ref_value).strip() != '' and str(ref_value) != 'N/A':
                            ref_values.append(str(ref_value))

                # Use the most common reference range or the first valid one
                if ref_values:
                    most_common_ref = Counter(ref_values).most_common(1)[0][0]
                    worksheet.write(row_num + 4, ref_col_idx, most_common_ref, ref_format)
                else:
                    worksheet.write(row_num + 4, ref_col_idx, '', ref_format)

            except Exception:
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
        for i, _col in enumerate(date_lab_cols_sorted):
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

                for col_idx, _date_lab_col in enumerate(date_lab_cols_sorted):
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

                    for _i, (date, value) in enumerate(zip(dates_for_charts, test_data, strict=False)):
                        if value is not None:  # Only include dates with actual values
                            valid_dates.append(date)
                            valid_values.append(value)
                            date_col_positions.append(chart_start_col + 1 + len(valid_dates) - 1)

                    # Write only the valid dates and values (no gaps)
                    for i, (date, value) in enumerate(zip(valid_dates, valid_values, strict=True)):
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
                        for col_idx, _col_name in enumerate(date_lab_cols_sorted):
                            if col_idx + 2 < len(ref_range_df.columns):
                                ref_value = ref_range_df.iloc[test_idx, col_idx + 2]
                                if pd.notna(ref_value) and str(ref_value).strip() != '' and str(ref_value) != 'N/A':
                                    ref_range = str(ref_value)
                                    break
                    except Exception:
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


def generate_pdf_health_report(df, patient_info, api_key):
    """
    Generate a PDF health report and return bytes.
    Uses ReportLab for reliable PDF generation without system dependencies.
    """
    from datetime import datetime
    from io import BytesIO

    from reportlab.lib import colors
    from reportlab.lib.colors import HexColor
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                           rightMargin=0.75*inch, leftMargin=0.75*inch,
                           topMargin=0.75*inch, bottomMargin=0.75*inch)

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle', parent=styles['Heading1'],
        fontSize=24, spaceAfter=30, textColor=HexColor('#1e293b'),
        alignment=1  # Center
    )
    heading_style = ParagraphStyle(
        'CustomHeading', parent=styles['Heading2'],
        fontSize=14, spaceAfter=12, textColor=HexColor('#6366f1'),
        spaceBefore=20
    )
    normal_style = ParagraphStyle(
        'CustomNormal', parent=styles['Normal'],
        fontSize=10, spaceAfter=6, textColor=HexColor('#374151')
    )

    story = []

    # Title
    story.append(Paragraph("Health Report", title_style))
    story.append(Paragraph(f"Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", normal_style))
    story.append(Spacer(1, 20))

    # Patient Info
    story.append(Paragraph("Patient Information", heading_style))
    patient_name = patient_info.get('name', 'N/A') if patient_info else 'N/A'
    patient_age = patient_info.get('age', 'N/A') if patient_info else 'N/A'
    patient_gender = patient_info.get('gender', 'N/A') if patient_info else 'N/A'
    story.append(Paragraph(f"<b>Name:</b> {patient_name}", normal_style))
    story.append(Paragraph(f"<b>Age:</b> {patient_age}", normal_style))
    story.append(Paragraph(f"<b>Gender:</b> {patient_gender}", normal_style))
    story.append(Spacer(1, 15))

    # Test Results Table
    story.append(Paragraph("Test Results Summary", heading_style))

    if df is not None and not df.empty:
        # Build table data
        table_data = [['Test Name', 'Value', 'Unit', 'Reference Range', 'Status']]

        for _, row in df.iterrows():
            test_name = str(row.get('Test_Name', row.get('test_name', 'N/A')))[:30]
            value = str(
                row.get('Result', row.get('result', row.get('Value', row.get('value', 'N/A'))))
            )
            unit = str(row.get('Unit', row.get('unit', '')))
            ref_range = str(row.get('Reference_Range', row.get('reference_range', 'N/A')))
            status = str(row.get('Status', row.get('status', 'Normal')))
            table_data.append([test_name, value, unit, ref_range, status])

        # Create table
        table = Table(table_data, colWidths=[2*inch, 0.8*inch, 0.6*inch, 1.5*inch, 0.8*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#6366f1')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#e5e7eb')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, HexColor('#f9fafb')]),
        ]))
        story.append(table)
    else:
        story.append(Paragraph("No test results available.", normal_style))

    story.append(Spacer(1, 20))

    # Disclaimer
    story.append(Paragraph("Disclaimer", heading_style))
    disclaimer_text = """This report is generated by an AI-powered analysis tool and is intended
    for informational purposes only. It does not constitute medical advice, diagnosis, or treatment.
    Always consult with qualified healthcare professionals for medical decisions."""
    story.append(Paragraph(disclaimer_text, normal_style))

    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
