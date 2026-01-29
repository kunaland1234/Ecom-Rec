"""
Data preprocessing pipeline for e-commerce recommendation system.

This module handles:
- Loading and cleaning event data
- Feature engineering from raw events
- Computing training statistics for normalization
- Handling train/test splits
"""

import pandas as pd
import numpy as np


def load_and_clean_events(path, cat_df, is_train=True, train_stats=None):
    """
    Load and preprocess event data with feature engineering.
    
    Args:
        path: Path to CSV file containing events
        cat_df: DataFrame with item->category mappings
        is_train: If True, compute training statistics. If False, use provided stats.
        train_stats: Dictionary of precomputed statistics (required if is_train=False)
    
    Returns:
        If is_train=True: (processed_df, train_stats)
        If is_train=False: processed_df
    """
    df = pd.read_csv(path)

    # Handle timestamp conversion properly
    if df["timestamp"].dtype == 'object':
        # Already datetime string from CSV
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    else:
        # Raw milliseconds
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    
    # Drop any invalid timestamps
    df = df.dropna(subset=["timestamp"])

    # Temporal features
    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["is_evening"] = ((df["hour"] >= 18) & (df["hour"] <= 23)).astype(int)
    df["is_working_hours"] = ((df["hour"] >= 9) & (df["hour"] <= 17)).astype(int)

    # Create target variable
    df["label"] = (df["event"] == "transaction").astype(int)

    # Merge category information
    df = df.merge(cat_df, on="itemid", how="left")
    df["category_id"] = df["category_id"].fillna(-1).astype(int)

    # Compute training statistics
    if is_train:
        train_stats = {}

        # Item popularity (total interactions per item)
        train_stats["item_popularity"] = (
            df.groupby("itemid").size().reset_index(name="item_popularity")
        )

        # Item conversion rate (purchases / total interactions)
        item_conv = df.groupby("itemid")["label"].agg(["sum", "count"]).reset_index()
        item_conv["item_conversion_rate"] = item_conv["sum"] / item_conv["count"]
        train_stats["item_conversion_rate"] = item_conv[
            ["itemid", "item_conversion_rate"]
        ]

        # Category popularity
        train_stats["category_popularity"] = (
            df.groupby("category_id").size().reset_index(name="category_popularity")
        )

        # Category conversion rate
        cat_conv = df.groupby("category_id")["label"].agg(["sum", "count"]).reset_index()
        cat_conv["category_conversion_rate"] = cat_conv["sum"] / cat_conv["count"]
        train_stats["category_conversion_rate"] = cat_conv[
            ["category_id", "category_conversion_rate"]
        ]

        # User activity level
        user_activity = (
            df.groupby("visitorid").size().reset_index(name="user_event_count")
        )
        train_stats["user_activity"] = user_activity

        # User purchase rate
        user_conv = df.groupby("visitorid")["label"].agg(["sum", "count"]).reset_index()
        user_conv["user_purchase_rate"] = user_conv["sum"] / user_conv["count"]
        train_stats["user_purchase_rate"] = user_conv[
            ["visitorid", "user_purchase_rate"]
        ]

        # Recent item popularity (last 7 days)
        max_ts = df["timestamp"].max()
        recent_cutoff = max_ts - pd.Timedelta(days=7)
        recent_df = df[df["timestamp"] >= recent_cutoff]
        train_stats["recent_item_popularity"] = (
            recent_df.groupby("itemid")
            .size()
            .reset_index(name="recent_item_popularity")
        )

        # Item popularity for ratio calculation
        train_stats["item_popularity_for_ratio"] = train_stats["item_popularity"]

        # Ensure all required stats exist (for edge cases)
        required = {
            "item_popularity": ["itemid", "item_popularity"],
            "item_conversion_rate": ["itemid", "item_conversion_rate"],
            "category_popularity": ["category_id", "category_popularity"],
            "category_conversion_rate": ["category_id", "category_conversion_rate"],
            "user_activity": ["visitorid", "user_event_count"],
            "user_purchase_rate": ["visitorid", "user_purchase_rate"],
            "recent_item_popularity": ["itemid", "recent_item_popularity"],
        }

        for key, cols in required.items():
            if key not in train_stats:
                train_stats[key] = pd.DataFrame(columns=cols)

    # Merge precomputed statistics
    df = df.merge(train_stats["item_popularity"], on="itemid", how="left")
    df["item_popularity"] = df["item_popularity"].fillna(0)

    df = df.merge(train_stats["item_conversion_rate"], on="itemid", how="left")
    df["item_conversion_rate"] = df["item_conversion_rate"].fillna(0)

    df = df.merge(train_stats["category_popularity"], on="category_id", how="left")
    df["category_popularity"] = df["category_popularity"].fillna(0)

    df = df.merge(train_stats["category_conversion_rate"], on="category_id", how="left")
    df["category_conversion_rate"] = df["category_conversion_rate"].fillna(0)

    df = df.merge(train_stats["user_activity"], on="visitorid", how="left")
    df["user_event_count"] = df["user_event_count"].fillna(0)

    df = df.merge(train_stats["user_purchase_rate"], on="visitorid", how="left")
    df["user_purchase_rate"] = df["user_purchase_rate"].fillna(0)

    # Sort by timestamp for sequential features
    df = df.sort_values("timestamp")

    # Sequential interaction features (cumulative counts)
    df["user_item_interaction_count"] = (
        df.groupby(["visitorid", "itemid"]).cumcount()
    )

    df["user_category_affinity"] = (
        df.groupby(["visitorid", "category_id"]).cumcount()
    )

    # Recent item popularity
    df = df.merge(train_stats["recent_item_popularity"], on="itemid", how="left")
    df["recent_item_popularity"] = df["recent_item_popularity"].fillna(0)

    # Item popularity ratio (how popular is this item compared to its category average?)
    category_avg_popularity = (
        train_stats["item_popularity_for_ratio"]
        .merge(df[["itemid", "category_id"]].drop_duplicates(), on="itemid")
        .groupby("category_id")["item_popularity"]
        .mean()
        .reset_index()
        .rename(columns={"item_popularity": "category_avg_item_popularity"})
    )

    df = df.merge(category_avg_popularity, on="category_id", how="left")
    df["item_popularity_ratio"] = (
        df["item_popularity"] / (df["category_avg_item_popularity"] + 1)
    ).fillna(0)

    # Select final feature set
    df = df[
        [
            "visitorid",
            "itemid",
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
            "label",
            "timestamp",
        ]
    ]

    if is_train:
        return df, train_stats
    else:
        return df


def load_item_categories(path1, path2):
    """
    Load and process item category mappings.
    
    Args:
        path1: Path to first item properties CSV
        path2: Path to second item properties CSV
    
    Returns:
        DataFrame with columns: itemid, category_id
    """
    p1 = pd.read_csv(path1)
    p2 = pd.read_csv(path2)

    items = pd.concat([p1, p2], ignore_index=True)
    cats = items[items["property"] == "categoryid"]

    cats = cats[["itemid", "value"]].drop_duplicates()
    cats = cats.rename(columns={"value": "category_id"})

    return cats