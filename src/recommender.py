"""
Recommendation model wrapper.

Handles model loading, scoring, and ranking of candidate items.
Supports LightGBM-only and optional MF hybrid scoring.

Strategy:
- For WARM users (in MF training set): Use hybrid LGB + MF
- For COLD users (not in MF training set): Use LGB only
"""

import joblib
import pandas as pd
import numpy as np


class Recommender:
    """
    Wrapper for the trained recommendation model.
    """

    def __init__(self, model_path, mf_model_path=None, alpha=0.3):
        """
        Args:
            model_path: Path to LightGBM model (.pkl)
            mf_model_path: Optional path to MF model (.pkl)
            alpha: Weight for MF score in hybrid scoring
        """
        # -------------------------
        # LightGBM (Primary model)
        # -------------------------
        artifact = joblib.load(model_path)
        self.model = artifact["model"]
        self.features = artifact["features"]

        # -------------------------
        # Matrix Factorization (Optional)
        # -------------------------
        self.alpha = alpha
        self.mf_model = None
        self.user_to_idx = None
        self.item_to_idx = None
        self.user_factors = None
        self.item_factors = None

        if mf_model_path is not None:
            try:
                mf_artifact = joblib.load(mf_model_path)
                self.mf_model = mf_artifact["model"]
                self.user_to_idx = mf_artifact["user_to_idx"]
                self.item_to_idx = mf_artifact["item_to_idx"]
                
                # Extract factor matrices directly for manual computation
                if hasattr(self.mf_model, 'A_') and self.mf_model.A_ is not None:
                    self.user_factors = self.mf_model.A_
                if hasattr(self.mf_model, 'B_') and self.mf_model.B_ is not None:
                    self.item_factors = self.mf_model.B_
                
                print(f"✅ MF model loaded successfully (alpha={alpha})")
                print(f"   Users: {len(self.user_to_idx):,}, Items: {len(self.item_to_idx):,}")
                print(f"   User factors shape: {self.user_factors.shape if self.user_factors is not None else 'N/A'}")
                print(f"   Item factors shape: {self.item_factors.shape if self.item_factors is not None else 'N/A'}")
                
            except Exception as e:
                print(f"⚠️  Could not load MF model: {e}")
                self.mf_model = None

    # ------------------------------------------------------------------
    # Internal MF scoring using manual dot product
    # ------------------------------------------------------------------
    def _mf_score_manual(self, user_id, item_id):
        """
        Compute MF score manually using factor matrices.
        This bypasses the broken predict() method in cmfrec.
        
        Returns:
            Score (float) or None if unavailable
        """
        if self.user_factors is None or self.item_factors is None:
            return None
        
        if user_id not in self.user_to_idx:
            return None
        
        if item_id not in self.item_to_idx:
            return None
        
        try:
            u_idx = self.user_to_idx[user_id]
            i_idx = self.item_to_idx[item_id]
            
            # Manual dot product: user_vector · item_vector
            user_vec = self.user_factors[u_idx]
            item_vec = self.item_factors[i_idx]
            
            score = np.dot(user_vec, item_vec)
            
            # Check if score is valid
            if np.isnan(score) or np.isinf(score):
                return None
            
            return float(score)
            
        except Exception as e:
            return None

    # ------------------------------------------------------------------
    # Batch MF scoring (more efficient)
    # ------------------------------------------------------------------
    def _mf_score_batch(self, user_id, item_ids):
        """
        Get MF scores for multiple items for a single user.
        Uses manual computation with factor matrices.
        
        Returns:
            List of scores (None for unavailable items)
        """
        if self.user_factors is None or self.item_factors is None:
            return [None] * len(item_ids)
        
        if user_id not in self.user_to_idx:
            return [None] * len(item_ids)
        
        try:
            u_idx = self.user_to_idx[user_id]
            user_vec = self.user_factors[u_idx]
            
            scores = []
            for item_id in item_ids:
                if item_id not in self.item_to_idx:
                    scores.append(None)
                    continue
                
                try:
                    i_idx = self.item_to_idx[item_id]
                    item_vec = self.item_factors[i_idx]
                    
                    score = np.dot(user_vec, item_vec)
                    
                    if np.isnan(score) or np.isinf(score):
                        scores.append(None)
                    else:
                        scores.append(float(score))
                        
                except Exception:
                    scores.append(None)
            
            return scores
            
        except Exception:
            return [None] * len(item_ids)

    # ------------------------------------------------------------------
    # Candidate scoring
    # ------------------------------------------------------------------
    def score_candidates(self, df, user_id=None):
        """
        Score candidate items.

        Args:
            df: DataFrame containing candidate items and features
                Must include 'itemid' and all model features
            user_id: Optional user id (required only for MF hybrid)

        Returns:
            DataFrame with final 'score' column
        """
        # Create output dataframe
        out = df.copy()
        
        # LightGBM probability
        X = df[self.features]
        lgb_scores = self.model.predict_proba(X)[:, 1]
        
        out["lgb_score"] = lgb_scores
        out["mf_score"] = None  # Initialize MF score column
        out["score"] = lgb_scores  # Default to LGB scores

        # -------------------------
        # Hybrid MF adjustment (only for warm users)
        # -------------------------
        if user_id is not None and self.user_factors is not None:
            # Check if user is in MF model (warm start)
            if user_id in self.user_to_idx:
                # Get MF scores for all items (batch mode for efficiency)
                mf_scores = self._mf_score_batch(user_id, out["itemid"].tolist())
                out["mf_score"] = mf_scores

                # Combine scores where MF is available
                mask = out["mf_score"].notna()
                
                if mask.any():
                    # Create final scores array
                    final_scores = lgb_scores.copy()
                    
                    # Hybrid score = (1-alpha)*LGB + alpha*MF
                    # Only combine where MF scores are available
                    mf_vals = out.loc[mask, "mf_score"].values
                    lgb_vals = lgb_scores[mask]
                    
                    final_scores[mask] = (1 - self.alpha) * lgb_vals + self.alpha * mf_vals
                    
                    out["score"] = final_scores
                    
                    # Log statistics
                    num_mf_scores = mask.sum()
                    print(f"   ✅ WARM USER: MF scores available for {num_mf_scores}/{len(out)} items ({100*num_mf_scores/len(out):.1f}%)")
                else:
                    print(f"   ⚠️  User in MF but no item scores available, using LGB only")
            else:
                print(f"   ℹ️  COLD START USER: Using LGB only (user not in MF training set)")

        return out

    # ------------------------------------------------------------------
    # Top-N ranking
    # ------------------------------------------------------------------
    def top_n(self, scored_df, n=5):
        """
        Return top-N items by final score.
        """
        return (
            scored_df
            .sort_values("score", ascending=False)
            .head(n)[["itemid", "score", "lgb_score", "mf_score"]]
        )