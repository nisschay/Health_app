"""
Report Generation Module
Handles PDF and Excel report generation
"""

import io
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional
from .data_processor import DataProcessor
from .visualization import VisualizationManager


class ReportGenerator:
    """
    Generates downloadable reports (PDF, Excel, CSV).
    Provides comprehensive health reports with charts and insights.
    """
    
    @classmethod
    def generate_pdf_report(cls, df: pd.DataFrame, patient_info: Dict, 
                           api_key: str = None) -> Optional[bytes]:
        """
        Generate a PDF health report using ReportLab.
        Returns PDF bytes or None on error.
        """
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.colors import HexColor
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
            from reportlab.lib.units import inch
            from reportlab.lib import colors
        except ImportError:
            return None
        
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer, pagesize=letter,
            rightMargin=0.75*inch, leftMargin=0.75*inch,
            topMargin=0.75*inch, bottomMargin=0.75*inch
        )
        
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle', parent=styles['Heading1'],
            fontSize=24, spaceAfter=30, textColor=HexColor('#1e293b'),
            alignment=1
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
        story.append(Paragraph(
            f"Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", 
            normal_style
        ))
        story.append(Spacer(1, 20))
        
        # Patient Info
        story.append(Paragraph("Patient Information", heading_style))
        story.append(Paragraph(f"<b>Name:</b> {patient_info.get('name', 'N/A')}", normal_style))
        story.append(Paragraph(f"<b>Age:</b> {patient_info.get('age', 'N/A')}", normal_style))
        story.append(Paragraph(f"<b>Gender:</b> {patient_info.get('gender', 'N/A')}", normal_style))
        story.append(Spacer(1, 15))
        
        # Health Score
        if not df.empty:
            health_data = VisualizationManager.calculate_health_score(df)
            story.append(Paragraph("Health Summary", heading_style))
            story.append(Paragraph(
                f"<b>Overall Health Score:</b> {health_data['overall_score']}/100", 
                normal_style
            ))
            story.append(Paragraph(
                f"<b>Total Tests:</b> {len(df)}", 
                normal_style
            ))
            story.append(Paragraph(
                f"<b>Tests Requiring Attention:</b> {len(health_data['concerns'])}", 
                normal_style
            ))
            story.append(Spacer(1, 15))
        
        # Test Results Table
        story.append(Paragraph("Test Results Summary", heading_style))
        
        if not df.empty:
            table_data = [['Test Name', 'Value', 'Unit', 'Reference', 'Status']]
            
            for _, row in df.head(50).iterrows():  # Limit to 50 rows
                table_data.append([
                    str(row.get('Test_Name', 'N/A'))[:25],
                    str(row.get('Result', 'N/A')),
                    str(row.get('Unit', '')),
                    str(row.get('Reference_Range', 'N/A'))[:15],
                    str(row.get('Status', 'Normal'))
                ])
            
            table = Table(table_data, colWidths=[2*inch, 0.8*inch, 0.5*inch, 1.2*inch, 0.7*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), HexColor('#6366f1')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 9),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
                ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#e5e7eb')),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, HexColor('#f9fafb')]),
            ]))
            story.append(table)
        else:
            story.append(Paragraph("No test results available.", normal_style))
        
        story.append(Spacer(1, 20))
        
        # Disclaimer
        story.append(Paragraph("Disclaimer", heading_style))
        story.append(Paragraph(
            "This report is for informational purposes only and does not constitute "
            "medical advice. Always consult with healthcare professionals.",
            normal_style
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
