import joblib

class Recommender:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)

    def score_candidates(self, df):
       
       
        features = df[["category_id", "hour", "dayofweek"]]
        df["score"] = self.model.predict_proba(features)[:, 1]
        return df

    def top_n(self, scored_df, n=5):
      
        return (
            scored_df
            .sort_values("score", ascending=False)
            .head(n)[["itemid", "score"]]
        )
