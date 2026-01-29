"""
Recommendation model wrapper.

Handles model loading, scoring, and ranking of candidate items.
"""

import joblib
import pandas as pd


class Recommender:
    """
    Wrapper for the trained recommendation model.
    
    Attributes:
        model: Trained LightGBM classifier
        features: List of feature names (must match training)
    """
    
    def __init__(self, model_path):
        """
        Load the trained model from disk.
        
        Args:
            model_path: Path to the saved model artifact (.pkl file)
        """
        artifact = joblib.load(model_path)
        self.model = artifact["model"]
        self.features = artifact["features"]

    def score_candidates(self, df):
        """
        Score candidate items using the trained model.
        
        Args:
            df: DataFrame with candidate items and their features
                Must contain all columns in self.features
        
        Returns:
            DataFrame with original data plus 'score' column
        """
        # Extract features in the correct order
        X = df[self.features]
        
        # Get probability of positive class (purchase)
        scores = self.model.predict_proba(X)[:, 1]

        # Add scores to output
        out = df.copy()
        out["score"] = scores
        
        return out

    def top_n(self, scored_df, n=5):
        """
        Get top N recommendations by score.
        
        Args:
            scored_df: DataFrame with 'score' column (from score_candidates)
            n: Number of top items to return
        
        Returns:
            DataFrame with columns: itemid, score
            Sorted by score (descending)
        """
        return (
            scored_df
            .sort_values("score", ascending=False)
            .head(n)[["itemid", "score"]]
        )