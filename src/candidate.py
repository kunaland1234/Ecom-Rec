import pandas as pd

def build_popular_items(events_df, top_k=50):
    """
    Build popular items per category based on interaction count
    """

    popular = (
        events_df
        .groupby(["category_id", "itemid"])
        .size()
        .reset_index(name="interaction_count")
        .sort_values("interaction_count", ascending=False)
    )


    popular = popular.groupby("category_id").head(top_k)

    return popular
