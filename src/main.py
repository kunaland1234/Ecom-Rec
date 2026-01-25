import datetime
import os
from pathlib import Path

import boto3
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from src.preprocess import load_and_clean_events, load_item_categories
from src.candidate import build_popular_items
from src.recommender import Recommender


# -----------------------
# App
# -----------------------
app = FastAPI(title="E-Commerce Recommender")


# -----------------------
# Environment
# -----------------------
S3_BUCKET = os.getenv("S3_BUCKET")
DATA_SOURCE = os.getenv("DATA_SOURCE", "s3")

if not S3_BUCKET:
    raise RuntimeError("S3_BUCKET environment variable not set")


# -----------------------
# AWS
# -----------------------
s3 = boto3.client("s3")


def download_from_s3(s3_key: str, local_path: str):
    """
    Download file from S3 only if it does not already exist locally.
    """
    local_path = Path(local_path)
    if not local_path.exists():
        s3.download_file(S3_BUCKET, s3_key, str(local_path))


# -----------------------
# Globals (loaded at startup)
# -----------------------
events = None
popular_items = None




# -----------------------
# Startup (CRITICAL)
# -----------------------
@app.on_event("startup")
def startup():
    global events, popular_items, recommender

    print("🚀 Starting recommender service (S3 mode)")

    # ---- Local paths
    p1 = "/tmp/item_properties_part1.csv"
    p2 = "/tmp/item_properties_part2.csv"
    events_path = "/tmp/events.csv"
    model_path = "/tmp/model_v2.pkl"

    # ---- Download data
    download_from_s3("data/item_properties_part1.csv", p1)
    download_from_s3("data/item_properties_part2.csv", p2)
    download_from_s3("data/events.csv", events_path)

    # ---- Preprocess
    cat_df = load_item_categories(p1, p2)
    events = load_and_clean_events(events_path, cat_df)

    # ---- Candidate generation
    popular_items = build_popular_items(events, top_k=50)

    # ---- Load model (ONLY ONE)
    download_from_s3("models/model_v2.pkl", model_path)
    recommender = Recommender(model_path)

    print("✅ Startup complete — recommender ready")


# -----------------------
# Health
# -----------------------
@app.get("/")
def health():
    return {"status": "ok"}


# -----------------------
# Recommend
# -----------------------
@app.get("/recommend/{user_id}")
def recommend(user_id: int, n: int = 5):
    if events is None or popular_items is None or recommender is None:
        raise HTTPException(status_code=503, detail="Service not ready")

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
