"""
Feature builder for real-time recommendation scoring.

FIXED: Handles cases where candidate items may not exist in the preprocessed data.
"""

import pandas as pd
import numpy as np


def build_features(df, visitorid, itemids):
    """
    Build feature vectors for scoring candidate items for a specific user.
    
    IMPORTANT: This must match the exact feature engineering logic used during training.
    
    Args:
        df: Preprocessed events DataFrame (with all precomputed features)
        visitorid: User ID to generate recommendations for
        itemids: List of candidate item IDs to score
    
    Returns:
        DataFrame with features for each candidate item, including 'itemid' column
    """
    # Use the latest timestamp in the dataset as reference point
    reference_time = df['timestamp'].max()

    # Get base features for candidate items (these are precomputed during preprocessing)
    # CRITICAL: Use the latest occurrence of each item to get most recent features
    base = (
        df[df["itemid"].isin(itemids)]
        .sort_values('timestamp', ascending=False)  # Get latest occurrence
        .drop_duplicates("itemid", keep='first')    # Keep most recent
        [[
            "itemid",
            "category_id",
            "item_popularity",
            "item_conversion_rate",
            "recent_item_popularity",
            "item_popularity_ratio",
            "category_popularity",
            "category_conversion_rate",
        ]]
        .copy()
    )
    
    # Check if all candidate items were found
    found_items = set(base['itemid'])
    missing_items = set(itemids) - found_items
    
    if missing_items:
        print(f"   ⚠️  Warning: {len(missing_items)} items not found in preprocessed data")
        # Create placeholder rows for missing items with default values
        missing_rows = []
        for item_id in missing_items:
            missing_rows.append({
                'itemid': item_id,
                'category_id': -1,  # Unknown category
                'item_popularity': 0,
                'item_conversion_rate': 0,
                'recent_item_popularity': 0,
                'item_popularity_ratio': 1.0,
                'category_popularity': 0,
                'category_conversion_rate': 0
            })
        
        if missing_rows:
            missing_df = pd.DataFrame(missing_rows)
            base = pd.concat([base, missing_df], ignore_index=True)

    # Get user's historical interactions
    user_df = df[df["visitorid"] == visitorid]

    # ===================================================================
    # TEMPORAL FEATURES
    # Based on the reference time (latest timestamp in dataset)
    # ===================================================================
    base["hour"] = reference_time.hour
    base["dayofweek"] = reference_time.weekday()
    base["is_weekend"] = int(reference_time.weekday() >= 5)
    base["is_evening"] = (1 if 18 <= reference_time.hour <= 23 else 0)
    base["is_working_hours"] = (1 if 9 <= reference_time.hour <= 17 else 0)

    # ===================================================================
    # USER-LEVEL FEATURES
    # Aggregated statistics about the user's behavior
    # ===================================================================
    if not user_df.empty:
        # Total number of events for this user
        base["user_event_count"] = len(user_df)
        
        # User's purchase rate (conversions / total interactions)
        user_purchase_count = user_df["label"].sum()
        base["user_purchase_rate"] = user_purchase_count / len(user_df)
    else:
        # New user with no history
        base["user_event_count"] = 0
        base["user_purchase_rate"] = 0

    # ===================================================================
    # USER-ITEM INTERACTION FEATURES
    # How has this user interacted with these specific items/categories?
    # ===================================================================
    # Count of past interactions between user and each candidate item
    item_counts = user_df["itemid"].value_counts()
    base["user_item_interaction_count"] = base["itemid"].map(item_counts).fillna(0)

    # Count of past interactions between user and each candidate item's category
    category_counts = user_df["category_id"].value_counts()
    base["user_category_affinity"] = base["category_id"].map(category_counts).fillna(0)

    # ===================================================================
    # FEATURE ORDERING
    # Must match training exactly - order matters for some models!
    # ===================================================================
    FEATURES = [
        "category_id",
        "hour",
        "dayofweek",
        "is_weekend",
        "is_evening",
        "is_working_hours",
        "item_popularity",
        "item_conversion_rate",
        "recent_item_popularity",
        "item_popularity_ratio",
        "category_popularity",
        "category_conversion_rate",
        "user_event_count",
        "user_purchase_rate",
        "user_item_interaction_count",
        "user_category_affinity",
    ]

    # Create output DataFrame with features + itemid
    X = base[FEATURES + ["itemid"]].copy()

    return X