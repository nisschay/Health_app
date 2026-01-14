"""
Report Generation Module
Handles PDF and Excel report generation with detailed health insights
"""

import io
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from .data_processor import DataProcessor
from .visualization import VisualizationManager


class ReportGenerator:
    """
    Generates downloadable reports (PDF, Excel, CSV).
    Provides comprehensive health reports with detailed analysis and insights.
    """
    
    # Status color mapping
    STATUS_COLORS = {
        'Normal': '#22c55e',
        'High': '#ef4444', 
        'Low': '#f97316',
        'Critical': '#dc2626',
        'N/A': '#6b7280'
    }
    
    @classmethod
    def _categorize_tests(cls, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Group tests by category for organized display"""
        if df.empty:
            return {}
        
        categories = {}
        for category in df['Test_Category'].unique():
            if pd.notna(category) and category != 'N/A':
                cat_df = df[df['Test_Category'] == category].copy()
                categories[category] = cat_df
        
        return dict(sorted(categories.items()))
    
    @classmethod
    def _get_status_indicator(cls, status: str) -> Tuple[str, str]:
        """Get emoji and color for status"""
        status_map = {
            'Normal': ('✓', '#22c55e'),
            'High': ('↑', '#ef4444'),
            'Low': ('↓', '#f97316'),
            'Critical': ('⚠', '#dc2626'),
            'Positive': ('⊕', '#ef4444'),
            'Negative': ('⊖', '#22c55e'),
            'N/A': ('—', '#6b7280')
        }
        return status_map.get(status, ('—', '#6b7280'))
    
    @classmethod
    def generate_pdf_report(cls, df: pd.DataFrame, patient_info: Dict, 
                           api_key: str = None) -> Optional[bytes]:
        """
        Generate a comprehensive PDF health report using ReportLab.
        Includes detailed test results organized by category with clear status indicators.
        """
        try:
            from reportlab.lib.pagesizes import letter, A4
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.colors import HexColor, Color
            from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, 
                                           Table, TableStyle, PageBreak, KeepTogether)
            from reportlab.lib.units import inch, cm
            from reportlab.lib import colors
            from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
        except ImportError:
            return None
        
        if df.empty:
            return None
        
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer, pagesize=A4,
            rightMargin=0.6*inch, leftMargin=0.6*inch,
            topMargin=0.6*inch, bottomMargin=0.6*inch
        )
        
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle', parent=styles['Heading1'],
            fontSize=22, spaceAfter=20, textColor=HexColor('#1e3a5f'),
            alignment=TA_CENTER, fontName='Helvetica-Bold'
        )
        
        subtitle_style = ParagraphStyle(
            'Subtitle', parent=styles['Normal'],
            fontSize=11, spaceAfter=15, textColor=HexColor('#64748b'),
            alignment=TA_CENTER
        )
        
        section_style = ParagraphStyle(
            'SectionHeading', parent=styles['Heading2'],
            fontSize=14, spaceAfter=10, spaceBefore=20,
            textColor=HexColor('#1e3a5f'), fontName='Helvetica-Bold',
            borderColor=HexColor('#3b82f6'), borderWidth=0,
            borderPadding=5
        )
        
        category_style = ParagraphStyle(
            'CategoryHeading', parent=styles['Heading3'],
            fontSize=12, spaceAfter=8, spaceBefore=15,
            textColor=HexColor('#6366f1'), fontName='Helvetica-Bold'
        )
        
        normal_style = ParagraphStyle(
            'CustomNormal', parent=styles['Normal'],
            fontSize=10, spaceAfter=4, textColor=HexColor('#374151')
        )
        
        small_style = ParagraphStyle(
            'SmallText', parent=styles['Normal'],
            fontSize=8, textColor=HexColor('#6b7280')
        )
        
        story = []
        
        # ===== HEADER =====
        story.append(Paragraph("🏥 Comprehensive Health Report", title_style))
        story.append(Paragraph(
            f"Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", 
            subtitle_style
        ))
        story.append(Spacer(1, 15))
        
        # ===== PATIENT INFO BOX =====
        patient_name = patient_info.get('name', 'N/A')
        patient_age = patient_info.get('age', 'N/A')
        patient_gender = patient_info.get('gender', 'N/A')
        patient_id = patient_info.get('patient_id', 'N/A')
        report_date = patient_info.get('date', 'N/A')
        lab_name = patient_info.get('lab_name', 'N/A')
        
        patient_table_data = [
            ['Patient Name', patient_name, 'Patient ID', patient_id],
            ['Age', patient_age, 'Gender', patient_gender],
            ['Report Date', report_date, 'Laboratory', lab_name[:30] if lab_name != 'N/A' else 'N/A']
        ]
        
        patient_table = Table(patient_table_data, colWidths=[1.2*inch, 2*inch, 1.2*inch, 2*inch])
        patient_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), HexColor('#f1f5f9')),
            ('BACKGROUND', (2, 0), (2, -1), HexColor('#f1f5f9')),
            ('TEXTCOLOR', (0, 0), (0, -1), HexColor('#475569')),
            ('TEXTCOLOR', (2, 0), (2, -1), HexColor('#475569')),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (2, 0), (2, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#e2e8f0')),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(patient_table)
        story.append(Spacer(1, 20))
        
        # ===== HEALTH SUMMARY =====
        health_data = VisualizationManager.calculate_health_score(df)
        score = health_data['overall_score']
        concerns = health_data['concerns']
        
        # Score indicator
        if score >= 80:
            score_color = '#22c55e'
            score_label = 'Good'
        elif score >= 60:
            score_color = '#f59e0b'
            score_label = 'Fair'
        else:
            score_color = '#ef4444'
            score_label = 'Needs Attention'
        
        story.append(Paragraph("📊 Health Summary", section_style))
        
        summary_data = [
            ['Overall Health Score', f'{score}/100 ({score_label})', 
             'Total Tests', str(len(df))],
            ['Tests Normal', str(len(df) - len(concerns)), 
             'Tests Requiring Attention', str(len(concerns))]
        ]
        
        summary_table = Table(summary_data, colWidths=[1.5*inch, 1.8*inch, 1.7*inch, 1.4*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), HexColor('#f8fafc')),
            ('TEXTCOLOR', (0, 0), (-1, -1), HexColor('#1e293b')),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (2, 0), (2, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#e2e8f0')),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(summary_table)
        story.append(Spacer(1, 15))
        
        # ===== ABNORMAL RESULTS SECTION =====
        if concerns:
            story.append(Paragraph("⚠️ Tests Requiring Attention", section_style))
            
            concern_header = [['Test Name', 'Result', 'Reference Range', 'Status', 'Test Date']]
            concern_rows = []
            
            for concern in concerns[:20]:  # Limit to 20
                status = concern.get('status', 'N/A')
                indicator, color = cls._get_status_indicator(status)
                test_date = concern.get('date', 'N/A')
                if test_date == 'N/A' or pd.isna(test_date):
                    test_date = 'N/A'
                
                concern_rows.append([
                    str(concern.get('test_name', 'N/A'))[:30],
                    str(concern.get('result', 'N/A')),
                    str(concern.get('reference', 'N/A'))[:20],
                    f"{indicator} {status}",
                    str(test_date)[:12]
                ])
            
            concern_table = Table(concern_header + concern_rows, 
                                 colWidths=[2*inch, 0.9*inch, 1.3*inch, 0.9*inch, 1.1*inch])
            concern_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), HexColor('#fef2f2')),
                ('TEXTCOLOR', (0, 0), (-1, 0), HexColor('#991b1b')),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('ALIGN', (3, 0), (3, -1), 'CENTER'),
                ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#fecaca')),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [HexColor('#ffffff'), HexColor('#fef2f2')]),
                ('TOPPADDING', (0, 0), (-1, -1), 5),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ]))
            story.append(concern_table)
        else:
            story.append(Paragraph("✅ All test results are within normal ranges!", 
                                  ParagraphStyle('Good', parent=normal_style, 
                                                textColor=HexColor('#22c55e'))))
        
        story.append(Spacer(1, 20))
        
        # ===== DETAILED RESULTS BY CATEGORY =====
        story.append(Paragraph("📋 Complete Test Results", section_style))
        story.append(Paragraph(
            "All test results organized by medical category with reference ranges and status indicators.",
            small_style
        ))
        story.append(Spacer(1, 10))
        
        categories = cls._categorize_tests(df)
        
        for category, cat_df in categories.items():
            # Category header
            story.append(Paragraph(f"▸ {category}", category_style))
            
            # Table header with Date column
            table_header = [['Test Name', 'Value', 'Unit', 'Reference', 'Status', 'Date']]
            table_rows = []
            
            for _, row in cat_df.iterrows():
                test_name = str(row.get('Test_Name', 'N/A'))
                result = str(row.get('Result', 'N/A'))
                unit = str(row.get('Unit', ''))
                ref_range = str(row.get('Reference_Range', 'N/A'))
                status = str(row.get('Status', 'N/A'))
                test_date = str(row.get('Test_Date', 'N/A'))
                if test_date == 'nan' or test_date == 'NaT':
                    test_date = 'N/A'
                
                indicator, _ = cls._get_status_indicator(status)
                
                table_rows.append([
                    test_name[:30],
                    result,
                    unit,
                    ref_range[:18],
                    f"{indicator} {status}",
                    test_date[:12]
                ])
            
            # Create table with adjusted column widths for date
            cat_table = Table(table_header + table_rows,
                             colWidths=[1.9*inch, 0.8*inch, 0.5*inch, 1.2*inch, 0.9*inch, 0.9*inch])
            
            # Style based on status
            table_style = [
                ('BACKGROUND', (0, 0), (-1, 0), HexColor('#6366f1')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 9),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('ALIGN', (4, 0), (4, -1), 'CENTER'),
                ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#e5e7eb')),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, HexColor('#f8fafc')]),
                ('TOPPADDING', (0, 0), (-1, -1), 5),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ]
            
            # Highlight abnormal rows
            for i, row in enumerate(table_rows, start=1):
                status_text = row[4]
                if '↑' in status_text or '↓' in status_text or '⚠' in status_text:
                    table_style.append(('BACKGROUND', (0, i), (-1, i), HexColor('#fef3c7')))
            
            cat_table.setStyle(TableStyle(table_style))
            story.append(KeepTogether([cat_table]))
            story.append(Spacer(1, 10))
        
        # ===== FOOTER / DISCLAIMER =====
        story.append(Spacer(1, 30))
        story.append(Paragraph("Important Notice", section_style))
        
        disclaimer_text = """
        <b>Disclaimer:</b> This report is computer-generated for informational purposes only and 
        should not be considered as medical advice or diagnosis. The results shown are extracted 
        from uploaded medical reports and should be reviewed by qualified healthcare professionals.
        <br/><br/>
        <b>Next Steps:</b>
        <br/>• Share this report with your healthcare provider for professional interpretation
        <br/>• For any abnormal results, consult with your doctor promptly
        <br/>• Regular health checkups are recommended for ongoing monitoring
        """
        
        story.append(Paragraph(disclaimer_text, normal_style))
        story.append(Spacer(1, 20))
        story.append(Paragraph(
            f"Report generated by Medical Report Analyzer • {datetime.now().strftime('%Y')}",
            ParagraphStyle('Footer', parent=small_style, alignment=TA_CENTER)
        ))
        
        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()
    
    @classmethod
    def generate_excel_report(cls, df: pd.DataFrame, patient_info: Dict) -> Optional[bytes]:
        """
        Generate an Excel report with formatted data and charts.
        Returns Excel bytes or None on error.
        """
        try:
            import xlsxwriter
        except ImportError:
            return None
        
        if df.empty:
            return None
        
        buffer = io.BytesIO()
        
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            # Prepare organized data
            organized_df = cls._prepare_organized_data(df)
            
            if organized_df.empty:
                return None
            
            organized_df.to_excel(writer, index=False, sheet_name='Medical Data', startrow=3)
            
            workbook = writer.book
            worksheet = writer.sheets['Medical Data']
            
            # Title formatting
            title_format = workbook.add_format({
                'bold': True, 'font_size': 16, 'align': 'center',
                'bg_color': '#4472C4', 'font_color': 'white'
            })
            
            worksheet.merge_range(
                'A1:' + chr(65 + len(organized_df.columns) - 1) + '1',
                'Medical Test Results', title_format
            )
            
            # Patient info row
            info_format = workbook.add_format({'italic': True, 'font_size': 10})
            worksheet.write('A2', 
                f"Patient: {patient_info.get('name', 'N/A')} | "
                f"Age: {patient_info.get('age', 'N/A')} | "
                f"Gender: {patient_info.get('gender', 'N/A')}",
                info_format
            )
            
            # Header formatting
            header_format = workbook.add_format({
                'bold': True, 'bg_color': '#D9E2F3', 'border': 1, 'align': 'center'
            })
            
            for col_num, col_name in enumerate(organized_df.columns):
                worksheet.write(3, col_num, col_name, header_format)
            
            # Auto-fit columns
            for i, col in enumerate(organized_df.columns):
                max_len = max(
                    organized_df[col].astype(str).map(len).max(),
                    len(str(col))
                ) + 2
                worksheet.set_column(i, i, min(max_len, 30))
            
            # Freeze header row
            worksheet.freeze_panes(4, 0)
            
            # Summary sheet
            summary_sheet = workbook.add_worksheet('Summary')
            summary_sheet.write('A1', 'Report Summary', title_format)
            summary_sheet.write('A3', f"Total Tests: {len(df)}")
            summary_sheet.write('A4', f"Categories: {df['Test_Category'].nunique()}")
            summary_sheet.write('A5', f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        
        buffer.seek(0)
        return buffer.getvalue()
    
    @classmethod
    def _prepare_organized_data(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare DataFrame for Excel export with proper organization"""
        if df.empty:
            return pd.DataFrame()
        
        # Select relevant columns
        cols = ['Test_Category', 'Test_Name', 'Test_Date', 'Result', 
                'Unit', 'Reference_Range', 'Status', 'Lab_Name']
        
        existing_cols = [c for c in cols if c in df.columns]
        
        result_df = df[existing_cols].copy()
        result_df = result_df.sort_values(['Test_Category', 'Test_Name', 'Test_Date'])
        
        return result_df.reset_index(drop=True)
    
    @classmethod
    def generate_csv_report(cls, df: pd.DataFrame) -> bytes:
        """Generate a CSV export of the data"""
        if df.empty:
            return b""
        
        # Exclude computed columns
        exclude_cols = ['Result_Numeric', 'Test_Date_dt']
        export_cols = [c for c in df.columns if c not in exclude_cols]
        
        return df[export_cols].to_csv(index=False).encode('utf-8')
