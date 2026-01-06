"""
Visualization Module
Handles chart generation and health insights dashboard
"""

import pandas as pd
import plotly.graph_objs as go
from typing import Dict, List, Optional, Tuple
from .data_processor import DataProcessor


# Body part mappings
CATEGORY_TO_BODY_PARTS = {
    "Haematology": ["Blood", "Bone Marrow"],
    "Complete Blood Count": ["Blood"],
    "Liver Function Test": ["Liver", "Digestive System"],
    "Kidney Function Test": ["Kidneys", "Urinary System"],
    "Lipid Profile": ["Blood", "Cardiovascular System", "Heart"],
    "Thyroid Profile": ["Thyroid", "Endocrine System"],
    "Blood Sugar": ["Blood", "Endocrine System", "Metabolism"],
    "Electrolytes": ["Blood", "Kidneys", "Metabolism"],
    "Urinalysis": ["Kidneys", "Urinary System"],
    "Cardiac Markers": ["Heart", "Cardiovascular System"],
    "Vitamins": ["Blood", "Nutritional Status", "Metabolism"],
    "Hormone Profile": ["Endocrine System", "Metabolism"],
    "Other": ["General"]
}

BODY_PART_EMOJIS = {
    "Blood": "🩸",
    "Bone Marrow": "🦴",
    "Liver": "🫁",
    "Kidneys": "🫘",
    "Heart": "❤️",
    "Thyroid": "🔄",
    "Endocrine System": "⚡",
    "Immune System": "🛡️",
    "General": "📊"
}


class VisualizationManager:
    """
    Manages chart creation and health visualization.
    Provides clean, clinical visualizations with proper styling.
    """
    
    @classmethod
    def create_test_chart(cls, df: pd.DataFrame, test_name: str, 
                          selected_date: str = None) -> Optional[go.Figure]:
        """
        Create a chart for a specific test.
        Shows trend line for multiple dates, or gauge for single date.
        """
        test_data = df[df['Test_Name'] == test_name].copy()
        
        if selected_date and selected_date != "All Dates":
            test_data = test_data[test_data['Test_Date'] == selected_date]
        
        if test_data.empty:
            return None
        
        # Sort by date
        test_data = test_data.sort_values('Test_Date_dt')
        
        # Check if we should show trend or single value
        unique_dates = test_data['Test_Date_dt'].nunique()
        has_numeric = test_data['Result_Numeric'].notna().any()
        
        if not has_numeric:
            return None
        
        if unique_dates > 1 and selected_date == "All Dates":
            return cls._create_trend_chart(test_data, test_name)
        else:
            return cls._create_gauge_chart(test_data.iloc[-1], test_name)
    
    @classmethod
    def _create_trend_chart(cls, test_data: pd.DataFrame, test_name: str) -> go.Figure:
        """Create a line chart showing test values over time"""
        fig = go.Figure()
        
        # Only include rows with numeric results
        test_data = test_data[test_data['Result_Numeric'].notna()]
        
        # Format dates for display
        test_data['Date_Display'] = test_data['Test_Date_dt'].apply(
            lambda x: DataProcessor.format_date(x)
        )
        
        # Add main line trace
        fig.add_trace(go.Scatter(
            x=test_data['Date_Display'],
            y=test_data['Result_Numeric'],
            mode='lines+markers',
            name='Result',
            line=dict(width=3, color='#a855f7'),
            marker=dict(size=8)
        ))
        
        # Add reference range lines
        latest = test_data.iloc[-1]
        low_ref, high_ref, ref_type = DataProcessor.parse_reference_range(
            latest.get('Reference_Range', '')
        )
        
        if ref_type == "range" and low_ref and high_ref:
            fig.add_hline(y=high_ref, line_dash="dash", line_color="#ef4444",
                         annotation_text="Upper Limit")
            fig.add_hline(y=low_ref, line_dash="dash", line_color="#22c55e",
                         annotation_text="Lower Limit")
            fig.add_hrect(y0=low_ref, y1=high_ref, fillcolor="#22c55e", opacity=0.1)
        
        unit = latest.get('Unit', '')
        
        fig.update_layout(
            title_text=f"{test_name} Trend ({unit})",
            xaxis_title="Date",
            yaxis_title=f"Result ({unit})",
            height=400,
            margin=dict(l=20, r=20, t=60, b=80),
            showlegend=True,
            template="plotly_dark",
            xaxis=dict(tickangle=45)
        )
        
        return fig
    
    @classmethod
    def _create_gauge_chart(cls, test_row: pd.Series, test_name: str) -> go.Figure:
        """Create a gauge/bullet chart for a single test value"""
        result = test_row.get('Result_Numeric')
        if pd.isna(result):
            return None
        
        ref_range_str = test_row.get('Reference_Range', '')
        unit = test_row.get('Unit', '')
        status = test_row.get('Status', 'N/A')
        
        low_ref, high_ref, ref_type = DataProcessor.parse_reference_range(ref_range_str)
        
        # Determine axis range and color steps
        if ref_type == "range" and low_ref and high_ref:
            span = high_ref - low_ref
            axis_min = min(low_ref - span * 0.2, result - span * 0.2)
            axis_max = max(high_ref + span * 0.2, result + span * 0.2)
            steps = [
                {'range': [axis_min, low_ref], 'color': 'lightcoral'},
                {'range': [low_ref, high_ref], 'color': 'lightgreen'},
                {'range': [high_ref, axis_max], 'color': 'lightcoral'}
            ]
        elif ref_type == "less_than" and high_ref:
            axis_min = 0
            axis_max = max(high_ref * 1.5, result * 1.2)
            steps = [
                {'range': [0, high_ref], 'color': 'lightgreen'},
                {'range': [high_ref, axis_max], 'color': 'lightcoral'}
            ]
        else:
            span = abs(result * 0.5) if result != 0 else 10
            axis_min = result - span
            axis_max = result + span
            steps = [{'range': [axis_min, axis_max], 'color': 'lightblue'}]
        
        fig = go.Figure(go.Indicator(
            mode="number+gauge",
            value=result,
            domain={'x': [0.1, 1], 'y': [0.2, 0.9]},
            number={'suffix': f" {unit}", 'font': {'size': 20}, 'valueformat': '.2f'},
            title={
                'text': f"<br>{test_name}<br><span style='font-size:0.8em;color:gray'>Ref: {ref_range_str} | Status: {status}</span>",
                'font': {"size": 12}
            },
            gauge={
                'shape': "bullet",
                'axis': {'range': [axis_min, axis_max]},
                'threshold': {
                    'line': {'color': "red", 'width': 3},
                    'thickness': 0.75,
                    'value': result
                },
                'steps': steps,
                'bar': {'color': "rgba(0,0,0,0)"}
            }
        ))
        
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=50, b=10))
        
        return fig
    
    @classmethod
    def calculate_health_score(cls, df: pd.DataFrame) -> Dict:
        """
        Calculate overall health score based on test results.
        Returns score (0-100) and breakdown.
        """
        if df.empty:
            return {'overall_score': 0, 'category_scores': {}, 'concerns': []}
        
        status_weights = {
            'Normal': 100,
            'N/A': 80,
            'Low': 60,
            'High': 40,
            'Critical': 10,
            'Positive': 30,
            'Negative': 100
        }
        
        # Calculate overall score
        weighted_sum = 0
        total_tests = len(df)
        
        for status, count in df['Status'].value_counts().items():
            status_clean = str(status).strip().title()
            weight = status_weights.get(status_clean, 70)
            weighted_sum += weight * count
        
        overall_score = int(weighted_sum / total_tests) if total_tests > 0 else 0
        
        # Calculate by category
        category_scores = {}
        for category in df['Test_Category'].unique():
            if pd.isna(category) or category == 'N/A':
                continue
            
            cat_df = df[df['Test_Category'] == category]
            cat_total = len(cat_df)
            if cat_total == 0:
                continue
            
            cat_sum = 0
            for _, row in cat_df.iterrows():
                status = str(row.get('Status', 'N/A')).strip().title()
                cat_sum += status_weights.get(status, 70)
            
            abnormal = cat_df[cat_df['Status'].isin(['High', 'Low', 'Critical', 'Positive'])]
            
            category_scores[category] = {
                'score': int(cat_sum / cat_total),
                'total_tests': cat_total,
                'abnormal_count': len(abnormal)
            }
        
        # Identify concerns
        concerns = []
        abnormal_df = df[df['Status'].isin(['High', 'Low', 'Critical', 'Positive'])]
        for _, row in abnormal_df.iterrows():
            concerns.append({
                'test_name': row.get('Test_Name', 'Unknown'),
                'result': row.get('Result', 'N/A'),
                'status': row.get('Status', 'N/A'),
                'reference': row.get('Reference_Range', 'N/A'),
                'category': row.get('Test_Category', 'N/A'),
                'date': row.get('Test_Date', 'N/A')
            })
        
        return {
            'overall_score': overall_score,
            'category_scores': category_scores,
            'concerns': concerns
        }
    
    @classmethod
    def get_body_systems_analysis(cls, df: pd.DataFrame) -> List[Dict]:
        """
        Analyze health by body system.
        Only flags systems with >= 50% abnormal tests.
        """
        if df.empty:
            return []
        
        body_systems = {}
        
        for category in df['Test_Category'].unique():
            if pd.isna(category) or category == 'N/A':
                continue
            
            cat_df = df[df['Test_Category'] == category]
            body_parts = CATEGORY_TO_BODY_PARTS.get(category, ['General'])
            
            for body_part in body_parts:
                if body_part not in body_systems:
                    body_systems[body_part] = {
                        'tests': [],
                        'abnormal_count': 0,
                        'total_count': 0,
                        'categories': set()
                    }
                
                body_systems[body_part]['categories'].add(category)
                body_systems[body_part]['total_count'] += len(cat_df)
                
                abnormal = cat_df[cat_df['Status'].isin(['High', 'Low', 'Critical', 'Positive'])]
                body_systems[body_part]['abnormal_count'] += len(abnormal)
                
                for _, row in cat_df.iterrows():
                    body_systems[body_part]['tests'].append({
                        'name': row.get('Test_Name', 'Unknown'),
                        'result': row.get('Result', 'N/A'),
                        'status': row.get('Status', 'N/A'),
                        'category': category
                    })
        
        # Build result list
        result = []
        for system, data in body_systems.items():
            if data['total_count'] == 0:
                continue
            
            abnormal_ratio = data['abnormal_count'] / data['total_count']
            
            # Only flag if >= 50% abnormal
            concern_level = 'flagged' if abnormal_ratio >= 0.50 else 'normal'
            
            result.append({
                'system': system,
                'emoji': BODY_PART_EMOJIS.get(system, '📊'),
                'concern_level': concern_level,
                'abnormal_count': data['abnormal_count'],
                'total_count': data['total_count'],
                'abnormal_ratio': abnormal_ratio,
                'categories': list(data['categories']),
                'tests': data['tests']
            })
        
        # Sort by concern level then abnormal count
        result.sort(key=lambda x: (0 if x['concern_level'] == 'flagged' else 1, -x['abnormal_count']))
        
        return result
