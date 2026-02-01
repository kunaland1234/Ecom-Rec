def build_popular_items(events_df, user_id, max_candidates=200, item_to_idx=None):
    """
    Generate candidate items for a user based on popularity.

    Strategy:
    1. Exclude items the user has already purchased
    2. Include items the user has viewed but not purchased
    3. Fill remaining slots with globally popular items

    Args:
        events_df: Preprocessed events DataFrame
        user_id: User ID to generate candidates for
        max_candidates: Maximum number of candidate items to return
        item_to_idx: Optional dict of itemid -> index (from MF model)

    Returns:
        List of candidate item IDs
    """

    # 🔹 Filter to only items known to MF model (if provided)
    if item_to_idx is not None:
        available_items = set(item_to_idx.keys())
        events_df = events_df[events_df["itemid"].isin(available_items)]

    # Get user's interaction history
    user_events = events_df[events_df["visitorid"] == user_id]

    # Items the user has purchased (exclude these)
    purchased_items = set(
        user_events[user_events["label"] == 1]["itemid"].unique()
    )

    # Items the user has viewed but not purchased
    viewed_items = set(
        user_events[user_events["label"] == 0]["itemid"].unique()
    )

    # Start with viewed items
    candidates = list(viewed_items - purchased_items)

    # If we need more candidates, add globally popular items
    if len(candidates) < max_candidates:
        popular_items = (
            events_df.groupby("itemid")
            .size()
            .sort_values(ascending=False)
            .index
            .tolist()
        )

        for item in popular_items:
            if item not in purchased_items and item not in candidates:
                candidates.append(item)
                if len(candidates) >= max_candidates:
                    break

    return candidates[:max_candidates]
