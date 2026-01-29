"""
Candidate generation module.

Generates a pool of candidate items to score for each user.
Uses popularity-based filtering to reduce computational cost.
"""

import pandas as pd


def build_popular_items(df, visitorid, max_candidates=200):
    """
    Generate candidate items for a user based on popularity.
    
    Strategy:
    1. Exclude items the user has already purchased
    2. Include items the user has viewed but not purchased
    3. Fill remaining slots with globally popular items
    
    Args:
        df: Preprocessed events DataFrame
        visitorid: User ID to generate candidates for
        max_candidates: Maximum number of candidate items to return
    
    Returns:
        List of candidate item IDs
    """
    # Get user's interaction history
    user_events = df[df["visitorid"] == visitorid]
    
    # Items the user has purchased (exclude these)
    purchased_items = set(user_events[user_events["label"] == 1]["itemid"].unique())
    
    # Items the user has viewed but not purchased (high priority candidates)
    viewed_items = set(user_events[user_events["label"] == 0]["itemid"].unique())
    
    # Start with viewed items
    candidates = list(viewed_items - purchased_items)
    
    # If we need more candidates, add globally popular items
    if len(candidates) < max_candidates:
        # Get global item popularity (from all events)
        popular_items = (
            df.groupby("itemid")
            .size()
            .sort_values(ascending=False)
            .index
            .tolist()
        )
        
        # Add popular items that user hasn't purchased
        for item in popular_items:
            if item not in purchased_items and item not in candidates:
                candidates.append(item)
                if len(candidates) >= max_candidates:
                    break
    
    return candidates[:max_candidates]