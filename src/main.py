"""
FastAPI application for serving e-commerce recommendations (AWS compatible).

Loads data and model from S3 into /tmp and serves inference APIs.
"""

import os
from pathlib import Path

import boto3
import pandas as pd
from fastapi import FastAPI, HTTPException

from src.preprocess import load_and_clean_events, load_item_categories
from src.candidate import build_popular_items
from src.build_feature import build_features
from src.recommender import Recommender


# --------------------------------------------------
# App
# --------------------------------------------------

print("🔥 API VERSION: 30-JAN-2026 — MAJOR API CHANGE 🔥")

app = FastAPI(
    title="E-Commerce Recommendation API",
    description="Personalized product recommendations using machine learning",
    version="1.0.1"
)


# --------------------------------------------------
# Environment
# --------------------------------------------------
S3_BUCKET = os.getenv("S3_BUCKET")
MODEL_VERSION = os.getenv("MODEL_VERSION", "v3")

if not S3_BUCKET:
    raise RuntimeError("S3_BUCKET environment variable not set")


# --------------------------------------------------
# AWS Clients
# --------------------------------------------------
s3 = boto3.client("s3")


# --------------------------------------------------
# Globals
# --------------------------------------------------
events_df = None
train_stats = None
cat_df = None
recommender = None


# --------------------------------------------------
# Utilities
# --------------------------------------------------
def download_from_s3(s3_key: str, local_path: str):
    """
    Download file from S3 only if it does not exist locally.
    """
    local_path = Path(local_path)
    if not local_path.exists():
        local_path.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(S3_BUCKET, s3_key, str(local_path))


# --------------------------------------------------
# Startup Hook (runs once on container start)
# --------------------------------------------------
@app.on_event("startup")
def startup():
    global events_df, train_stats, cat_df, recommender

    # Local ECS-safe paths
    p1 = "/tmp/item_properties_part1.csv"
    p2 = "/tmp/item_properties_part2.csv"
    events_path = "/tmp/events.csv"
    model_path = f"/tmp/model_{MODEL_VERSION}.pkl"

    # Download from S3
    download_from_s3("data/item_properties_part1.csv", p1)
    download_from_s3("data/item_properties_part2.csv", p2)
    download_from_s3("data/events.csv", events_path)
    download_from_s3(f"models/model_{MODEL_VERSION}.pkl", model_path)

    # Load categories
    cat_df = load_item_categories(p1, p2)

    # Load & preprocess events (inference mode)
    events_df, train_stats = load_and_clean_events(
        events_path,
        cat_df,
        is_train=True
    )

    # Load model
    recommender = Recommender(model_path)


# --------------------------------------------------
# Endpoints
# --------------------------------------------------
@app.get("/")
def health_check():
    return {
        "status": "healthy",
        "model_version": MODEL_VERSION,
        "total_events": len(events_df) if events_df is not None else 0,
    }


@app.get("/recommend/{user_id}")
def recommend(user_id: int, n: int = 5):
    if events_df is None or recommender is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    if n < 1 or n > 50:
        raise HTTPException(status_code=400, detail="n must be between 1 and 50")

    if user_id not in events_df["visitorid"].values:
        raise HTTPException(status_code=404, detail="User not found")

    # Candidate generation
    candidate_itemids = build_popular_items(
        events_df,
        user_id,
        max_candidates=200
    )

    if len(candidate_itemids) == 0:
        raise HTTPException(status_code=404, detail="No candidates found")

    # Feature building
    feature_df = build_features(
        events_df,
        user_id,
        candidate_itemids
    )

    # Scoring
    scored = recommender.score_candidates(feature_df)
    top_n = recommender.top_n(scored, n=n)

    return {
        "user_id": user_id,
        "recommendations": top_n.to_dict(orient="records"),
        "total_candidates_scored": len(candidate_itemids)
    }


@app.get("/user/{user_id}/stats")
def user_stats(user_id: int):
    if events_df is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    if user_id not in events_df["visitorid"].values:
        raise HTTPException(status_code=404, detail="User not found")

    user_events = events_df[events_df["visitorid"] == user_id]

    return {
        "user_id": user_id,
        "total_events": int(len(user_events)),
        "purchases": int(user_events["label"].sum()),
        "conversion_rate": float(user_events["label"].mean()),
        "unique_items_viewed": int(user_events["itemid"].nunique()),
        "unique_categories_viewed": int(user_events["category_id"].nunique()),
        "first_seen": str(user_events["timestamp"].min()),
        "last_seen": str(user_events["timestamp"].max())
    }
