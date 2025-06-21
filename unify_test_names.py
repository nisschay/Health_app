import pandas as pd
from rapidfuzz import fuzz
from collections import Counter
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import re

# Improved: cluster similar names using token set fuzzy distances on refined normalized names

def _normalize_test_name(name):
    """
    Refine test name by lowercasing, removing parentheses content, common trailing words,
    punctuation (except spaces and hyphens), standardizing spaces/hyphens, and simple pluralization.
    """
    # Handle non-string inputs
    if not isinstance(name, str):
        return 'N/A'

    name = name.strip()
    # Lowercase
    name = name.lower()

    # Remove all content within parentheses, including the parentheses
    name = re.sub(r'\([^)]*\)', '', name)

    # Standardize spaces around hyphens (e.g., 'B - 12' -> 'B-12')
    name = re.sub(r'\s*-\s*', '-', name)

    # Remove common trailing words (like Level, Total, Count) using word boundaries
    trailing_words_to_remove = ['level', 'total', 'count', 'profile', 'panel', 'test', 'assay', 'value', 'concentration']
    for word in trailing_words_to_remove:
        name = re.sub(f'[\s-]*\\b{word}\\b$', '', name)

    # Remove all non-alphanumeric characters except spaces and hyphens
    name = re.sub(r'[^a-z0-9\s-]', '', name)

    # Replace multiple spaces with a single space and strip leading/trailing spaces/hyphens
    name = re.sub(r'\s+', ' ', name).strip(' -')

    # Simple plural normalization: if ends with 's', remove it (not perfect, but helps)
    if name.endswith('s') and not name.endswith('ss') and len(name) > 1:
        name = name[:-1]

    return name

def unify_test_names(df, threshold=90):
    """
    Unify similar test names in the DataFrame using refined normalization and token set fuzzy grouping.
    Args:
        df: DataFrame with a 'Test_Name' column
        threshold: Similarity threshold for grouping (0-100, higher is stricter)
    Returns:
        DataFrame with unified test names
    """
    # Handle NaN values in Test_Name before getting unique names
    df['Test_Name'] = df['Test_Name'].fillna('N/A')
    original_names = df['Test_Name'].unique().tolist()
    n = len(original_names)

    if n == 0:
        return df # Return empty df if no names

    # Create a mapping from original name to normalized name
    norm_map = {name: _normalize_test_name(name) for name in original_names}
    normalized_names = [norm_map[name] for name in original_names] # Ensure order matches original_names

    # Compute distance matrix using token_set_ratio on *normalized* names
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            # Use token_set_ratio on normalized names
            score = fuzz.token_set_ratio(normalized_names[i], normalized_names[j]) / 100.0
            dist[i, j] = dist[j, i] = 1 - score

    # Cluster names using agglomerative clustering
    threshold = max(0, min(100, threshold))
    distance_threshold_val = 1 - (threshold / 100.0)

    # Handle edge cases for clustering
    if n > 0 and np.all(dist <= distance_threshold_val):
         # All names should be in one cluster, pick the most common original name
         most_common_original_name = Counter(df['Test_Name']).most_common(1)[0][0]
         name_map = {name: most_common_original_name for name in original_names}
         df['Test_Name'] = df['Test_Name'].map(name_map)
         df = df.drop_duplicates(['Test_Name', 'Test_Date', 'Result', 'Patient_ID'])
         df = df.reset_index(drop=True)
         return df

    if n > 0 and np.all(dist > distance_threshold_val) and distance_threshold_val > 0:
        # No clusters formed beyond individual names, return original df after filling NaN and dropping duplicates
        # Create an identity map in this case
        name_map = {name: name for name in original_names}
        df['Test_Name'] = df['Test_Name'].map(name_map) # Apply identity map
        return df.drop_duplicates(['Test_Name', 'Test_Date', 'Result', 'Patient_ID']).reset_index(drop=True)

    # Need to handle the case where n=1 separately as fit_predict might behave unexpectedly or is unnecessary
    if n == 1:
        # Create an identity map for a single name
        name_map = {original_names[0]: original_names[0]}
        df['Test_Name'] = df['Test_Name'].map(name_map) # Apply identity map
        return df.drop_duplicates(['Test_Name', 'Test_Date', 'Result', 'Patient_ID']).reset_index(drop=True)

    clustering = AgglomerativeClustering(
        metric='precomputed',
        linkage='complete', # 'complete' or 'average' usually work well
        distance_threshold=distance_threshold_val,
        n_clusters=None
    )

    labels = clustering.fit_predict(dist)

    # Map each original name to the most common original name in its cluster
    name_map = {}
    labeled_original_names = list(zip(labels, original_names))

    for label in set(labels):
        cluster_original_names = [name for lbl, name in labeled_original_names if lbl == label]
        counts = Counter(df[df['Test_Name'].isin(cluster_original_names)]['Test_Name'])
        if counts:
             most_common_original_name = counts.most_common(1)[0][0]
        else:
             most_common_original_name = cluster_original_names[0] if cluster_original_names else 'N/A'
        for original_name in cluster_original_names:
            name_map[original_name] = most_common_original_name

    df['Test_Name'] = df['Test_Name'].map(name_map)
    df = df.drop_duplicates(['Test_Name', 'Test_Date', 'Result', 'Patient_ID'])
    df = df.reset_index(drop=True)
    return df
