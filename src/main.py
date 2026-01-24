import datetime
import pandas as pd
from fastapi import FastAPI, HTTPException

from src.preprocess import load_and_clean_events, load_item_categories
from src.candidate import build_popular_items
from src.recommender import Recommender


app = FastAPI(title="E-Commerce Recommender")

# ---------
# Load everything ONCE at startup
# ---------
cat_df = load_item_categories(
    "data/item_properties_part1.csv",
    "data/item_properties_part2.csv"
)

events = load_and_clean_events("data/events.csv", cat_df)
popular_items = build_popular_items(events, top_k=50)

recommender = Recommender("models/model_v2.pkl")


@app.get("/")
def health():
    return {"status": "ok"}


@app.get("/recommend/{user_id}")
def recommend(user_id: int, n: int = 5):
    user_events = events[events["visitorid"] == user_id]

    if user_events.empty:
        raise HTTPException(status_code=404, detail="User not found")

    last_event = user_events.iloc[-1]
    user_category = last_event["category_id"]

    candidates = popular_items[
        popular_items["category_id"] == user_category
    ].copy()

    if candidates.empty:
        raise HTTPException(status_code=404, detail="No candidates found")

    now = datetime.datetime.now()
    candidates["hour"] = now.hour
    candidates["dayofweek"] = now.weekday()

    scored = recommender.score_candidates(candidates)
    top_n = recommender.top_n(scored, n=n)

    return {
        "user_id": user_id,
        "recommendations": top_n.to_dict(orient="records")
    }
