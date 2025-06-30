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

    # --- Explicit mapping for specific names ---
    # First, find most common category for each test name
    test_categories = {}
    for test_name in df['Test_Name'].unique():
        categories = df[df['Test_Name'] == test_name]['Test_Category'].dropna()
        if not categories.empty:
            most_common_category = categories.mode().iloc[0]
            test_categories[test_name] = most_common_category

    # Create explicit mapping for vitamin B12 variants while preserving categories
    vitamin_b12_variants = [
        'Vitamin B-12', 'Vitamin B12', 'Vitamin B 12',
        'Vitamin B-12 Level', 'Vitamin B12 Level', 'Vitamin B 12 Level',
        'vitamin b-12', 'vitamin b12', 'vitamin b 12',
        'vitamin b-12 level', 'vitamin b12 level', 'vitamin b 12 level',
        'Vitamin B - 12', 'Vitamin B - 12 Level', # Added variants with spaces around dash
        'vitamin b - 12', 'vitamin b - 12 level' # Added lowercase variants with spaces around dash
    ]

    # Map all variants to 'Vitamin B12' and update their category if exists
    target_name = 'Vitamin B12'
    explicit_map = {v: target_name for v in vitamin_b12_variants}
    df['Test_Name'] = df['Test_Name'].replace(explicit_map)

    # After name replacement, update categories
    for old_name in vitamin_b12_variants:
        if old_name in test_categories:
            mask = df['Test_Name'] == target_name
            df.loc[mask & df['Test_Category'].isna(), 'Test_Category'] = test_categories[old_name]

    # For any remaining N/A categories, try to fill from existing data
    for test_name in df['Test_Name'].unique():
        mask = (df['Test_Name'] == test_name) & (df['Test_Category'].isna() | (df['Test_Category'] == 'N/A'))
        existing_categories = df[df['Test_Name'] == test_name]['Test_Category'].dropna()
        if not existing_categories.empty and any(mask):
            most_common_category = existing_categories[existing_categories != 'N/A'].mode().iloc[0] if not existing_categories[existing_categories != 'N/A'].empty else existing_categories.mode().iloc[0]
            df.loc[mask, 'Test_Category'] = most_common_category

    original_names = df['Test_Name'].unique().tolist()
    n = len(original_names)

    if n == 0:
        return df # Return empty df if no names

    # Create a mapping from original name to normalized name
    norm_map = {name: _normalize_test_name(name) for name in original_names}
    normalized_names = [norm_map[name] for name in original_names] # Ensure order matches original_names

    # Compute distance matrix using token_set_ratio on *normalized* names
    dist = np.zeros((n, n))

    # Cluster names using agglomerative clustering
    threshold = max(0, min(100, threshold))
    distance_threshold_val = 1 - (threshold / 100.0)

    for i in range(n):
        for j in range(i+1, n):
            name1 = original_names[i].lower()
            name2 = original_names[j].lower()

            # --- Custom logic to prevent merging specific cholesterol types ---
            prevent_merge = False
            is_chol1 = 'cholesterol' in name1
            is_chol2 = 'cholesterol' in name2
            is_hdl1 = 'hdl' in name1
            is_hdl2 = 'hdl' in name2
            is_ldl1 = 'ldl' in name1
            is_ldl2 = 'ldl' in name2
            is_vldl1 = 'vldl' in name1
            is_vldl2 = 'vldl' in name2
            is_total1 = 'total' in name1
            is_total2 = 'total' in name2

            # Prevent merging if one is HDL/LDL/VLDL and the other is generic/total cholesterol
            if (is_hdl1 or is_ldl1 or is_vldl1) and is_chol2 and not (is_hdl2 or is_ldl2 or is_vldl2):
                prevent_merge = True
            if (is_hdl2 or is_ldl2 or is_vldl2) and is_chol1 and not (is_hdl1 or is_ldl1 or is_vldl1):
                prevent_merge = True

            # Prevent merging between different lipoprotein variants
            if (is_hdl1 and (is_ldl2 or is_vldl2)) or (is_ldl1 and (is_hdl2 or is_vldl2)) or (is_vldl1 and (is_hdl2 or is_ldl2)):
                prevent_merge = True

            if prevent_merge:
                score = 0.0 # Maximum distance (1 - 0.0 = 1.0)
            else:
                # Use token_set_ratio on normalized names for other cases
                score = fuzz.token_set_ratio(normalized_names[i], normalized_names[j]) / 100.0

            dist[i, j] = dist[j, i] = 1 - score

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

    # --- Value-based cluster splitting ---
    # For each cluster, check if any names have median 'Result' values >20% off from the cluster median
    # If so, split them into separate subclusters
    name_map = {}
    labeled_original_names = list(zip(labels, original_names))
    used_label = max(labels) + 1
    for label in set(labels):
        cluster_original_names = [name for lbl, name in labeled_original_names if lbl == label]
        # Only consider clusters with more than 1 name
        if len(cluster_original_names) > 1:
            # Compute median Result for each name
            medians = {}
            for name in cluster_original_names:
                vals = pd.to_numeric(df[df['Test_Name'] == name]['Result'], errors='coerce')
                vals = vals.dropna()
                if len(vals) > 0:
                    medians[name] = vals.median()
                else:
                    medians[name] = np.nan
            # Compute cluster median (ignoring NaN)
            cluster_median = np.nanmedian(list(medians.values()))
            # Split names whose median is >30% off from cluster median
            for name in cluster_original_names:
                m = medians[name]
                if not np.isnan(m) and cluster_median > 0:
                    diff = abs(m - cluster_median) / cluster_median
                    if diff > 0.3:
                        # Assign a new label for this outlier
                        for i, (lbl, n) in enumerate(labeled_original_names):
                            if n == name:
                                labeled_original_names[i] = (used_label, n)
                        used_label += 1
        # else: keep as is

    # Now, re-map names to most common original name in their (possibly split) cluster
    for label in set(lbl for lbl, _ in labeled_original_names):
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
