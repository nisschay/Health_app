"""Rule-based summaries of a set of results: the health score and the body-system view."""
from __future__ import annotations

import pandas as pd

from .category_mapping import BODY_PARTS_TO_EMOJI, TEST_CATEGORY_TO_BODY_PARTS
from .normalization import (
    CONCERNING_STATUS_VALUES,
    STATUS_HEALTH_WEIGHTS,
    UNKNOWN_STATUS_HEALTH_WEIGHT,
    normalize_status,
)


def calculate_health_score(df):
    """
    Calculate an overall health score based on test results.
    Returns a score from 0-100 and breakdown by category.
    """
    if df.empty:
        return {'overall_score': 0, 'category_scores': {}, 'concerns': []}

    df = df.copy()
    df['Status'] = df['Status'].map(normalize_status)
    total_tests = len(df)

    weighted_sum = 0
    for status, count in df['Status'].value_counts().to_dict().items():
        weighted_sum += STATUS_HEALTH_WEIGHTS.get(status, UNKNOWN_STATUS_HEALTH_WEIGHT) * count

    overall_score = int(weighted_sum / total_tests) if total_tests > 0 else 0

    # Calculate scores by category
    category_scores = {}
    for category in df['Test_Category'].unique():
        if pd.isna(category) or category == 'N/A':
            continue
        cat_df = df[df['Test_Category'] == category]
        cat_total = len(cat_df)
        if cat_total == 0:
            continue

        cat_weighted_sum = sum(
            STATUS_HEALTH_WEIGHTS.get(status, UNKNOWN_STATUS_HEALTH_WEIGHT)
            for status in cat_df['Status']
        )

        category_scores[category] = {
            'score': int(cat_weighted_sum / cat_total),
            'total_tests': cat_total,
            'abnormal_count': len(cat_df[cat_df['Status'].isin(CONCERNING_STATUS_VALUES)])
        }

    # Identify concerns (tests with abnormal values)
    concerns = []
    abnormal_df = df[df['Status'].isin(CONCERNING_STATUS_VALUES)]
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


def get_body_system_analysis(df):
    """
    Analyze health by body system with concern levels.
    Returns sorted list from most concerning to least.
    """
    if df.empty:
        return []

    body_systems = {}

    df = df.copy()
    df['Status'] = df['Status'].map(normalize_status)

    for category in df['Test_Category'].unique():
        if pd.isna(category) or category == 'N/A':
            continue

        cat_df = df[df['Test_Category'] == category]

        # Get body parts for this category
        body_parts = TEST_CATEGORY_TO_BODY_PARTS.get(category, ['General'])

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

            abnormal = cat_df[cat_df['Status'].isin(CONCERNING_STATUS_VALUES)]
            body_systems[body_part]['abnormal_count'] += len(abnormal)

            for _, row in cat_df.iterrows():
                body_systems[body_part]['tests'].append({
                    'name': row.get('Test_Name', 'Unknown'),
                    'result': row.get('Result', 'N/A'),
                    'status': row.get('Status', 'N/A'),
                    'category': category
                })

    # Calculate concern level for each body system
    # CRITICAL: Only flag systems where >= 50% of tests are abnormal
    # This prevents over-alarming patients with misleading visual indicators
    result = []
    for system, data in body_systems.items():
        if data['total_count'] == 0:
            continue

        abnormal_ratio = data['abnormal_count'] / data['total_count']

        # MEDICAL LOGIC: Only flag if >= 50% abnormal (strict threshold)
        if abnormal_ratio >= 0.50:
            concern_level = 'flagged'
            concern_score = 2
        else:
            concern_level = 'normal'
            concern_score = 1

        result.append({
            'system': system,
            'emoji': BODY_PARTS_TO_EMOJI.get(system, '📊'),
            'concern_level': concern_level,
            'concern_score': concern_score,
            'abnormal_count': data['abnormal_count'],
            'total_count': data['total_count'],
            'abnormal_ratio': abnormal_ratio,
            'categories': list(data['categories']),
            'tests': data['tests']
        })

    # Sort by concern score (highest first), then by abnormal count
    result.sort(key=lambda x: (-x['concern_score'], -x['abnormal_count']))

    return result
