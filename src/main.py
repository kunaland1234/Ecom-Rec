"""
FastAPI application for serving e-commerce recommendations (AWS compatible).

Endpoints:
- GET /: Health check and system status
- GET /recommend/{user_id}: Get personalized recommendations for a user

Hybrid Model:
- Warm users (in MF training): LGB + MF hybrid scoring
- Cold users (not in MF training): LGB only

Loads data and model from S3 into /tmp and serves inference APIs.
"""

import os
from pathlib import Path
import logging

import boto3
import pandas as pd
from fastapi import FastAPI, HTTPException

from src.preprocess import load_and_clean_events, load_item_categories
from src.candidate import build_popular_items
from src.build_feature import build_features
from src.recommender import Recommender


# --------------------------------------------------
# Logging Configuration
# --------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --------------------------------------------------
# App
# --------------------------------------------------
print("API VERSION: HYBRID MODEL (LGB + MF)")

app = FastAPI(
    title="E-Commerce Recommendation API",
    description="Personalized product recommendations using hybrid ML (LightGBM + Matrix Factorization)",
    version="1.0.1"
)


# --------------------------------------------------
# Environment Variables
# --------------------------------------------------
S3_BUCKET = os.getenv("S3_BUCKET")
MODEL_VERSION = os.getenv("MODEL_VERSION", "v3")
MF_ALPHA = float(os.getenv("MF_ALPHA", "0.3"))  # Weight for MF score in hybrid

if not S3_BUCKET:
    raise RuntimeError("S3_BUCKET environment variable not set")


# --------------------------------------------------
# AWS Clients
# --------------------------------------------------
s3 = boto3.client("s3")


# --------------------------------------------------
# Global Variables
# --------------------------------------------------
events_df = None
train_stats = None
cat_df = None
recommender = None
artifact = None


# --------------------------------------------------
# Utilities
# --------------------------------------------------
def download_from_s3(s3_key: str, local_path: str):
    """
    Download file from S3 only if it does not exist locally.
    Uses /tmp directory for ECS compatibility.
    """
    local_path = Path(local_path)
    if not local_path.exists():
        local_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Downloading s3://{S3_BUCKET}/{s3_key} -> {local_path}")
        s3.download_file(S3_BUCKET, s3_key, str(local_path))
        logger.info(f"✅ Downloaded {s3_key}")
    else:
        logger.info(f"✅ File already exists: {local_path}")


# --------------------------------------------------
# Startup Hook (runs once on container start)
# --------------------------------------------------
@app.on_event("startup")
def startup():
    global events_df, train_stats, cat_df, recommender, artifact

    print("="*60)
    print("INITIALIZING RECOMMENDATION SERVICE (AWS MODE)")
    print("="*60)

    # Local ECS-safe paths (using /tmp for AWS compatibility)
    p1 = "/tmp/item_properties_part1.csv"
    p2 = "/tmp/item_properties_part2.csv"
    events_path = "/tmp/events.csv"
    model_path = f"/tmp/model_{MODEL_VERSION}.pkl"
    mf_model_path = "/tmp/mf_model.pkl"

    # Download data files from S3
    print("\n📦 Downloading data from S3...")
    download_from_s3("data/item_properties_part1.csv", p1)
    download_from_s3("data/item_properties_part2.csv", p2)
    download_from_s3("data/events.csv", events_path)
    
    # Download model files from S3
    print("\n🤖 Downloading models from S3...")
    download_from_s3(f"models/model_{MODEL_VERSION}.pkl", model_path)
    download_from_s3("models/mf_model.pkl", mf_model_path)

    # Load item categories
    print("\n📂 Loading item categories...")
    cat_df = load_item_categories(p1, p2)
    print(f"✅ Loaded {len(cat_df):,} item-category mappings")

    # Load raw events
    print("\n📊 Loading raw events...")
    df_raw = pd.read_csv(events_path)
    df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"], unit="ms")
    print(f"✅ Loaded {len(df_raw):,} events")
    print(f"   Date range: {df_raw['timestamp'].min()} to {df_raw['timestamp'].max()}")

    # Save to temporary file for preprocessing
    temp_file = "/tmp/temp_inference.csv"
    df_raw.to_csv(temp_file, index=False)

    # Load & preprocess events
    print("\n🔧 Preprocessing events...")
    events_df, train_stats = load_and_clean_events(
        temp_file,
        cat_df,
        is_train=True  # Set to True to generate train_stats
    )
    
    print(f"✅ Preprocessed {len(events_df):,} events")
    print(f"   Unique users: {events_df['visitorid'].nunique():,}")
    print(f"   Unique items: {events_df['itemid'].nunique():,}")

    # Clean up temporary file
    if os.path.exists(temp_file):
        os.remove(temp_file)

    # Load hybrid model (LGB + MF)
    print(f"\n🚀 Loading hybrid model from {model_path}...")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. Please ensure model is uploaded to S3."
        )

    # Initialize recommender with hybrid MF model
    recommender = Recommender(
        model_path, 
        mf_model_path=mf_model_path, 
        alpha=MF_ALPHA
    )
    
    # Load artifact for metadata
    import joblib
    artifact = joblib.load(model_path)
    
    print(f"✅ Model loaded successfully")
    print(f"   LGB features: {len(artifact['features'])}")
    print(f"   MF users: {len(recommender.user_to_idx) if recommender.user_to_idx else 0}")
    print(f"   MF items: {len(recommender.item_to_idx) if recommender.item_to_idx else 0}")
    print(f"   Hybrid alpha: {MF_ALPHA}")

    print("\n✅ SERVICE READY!")
    print("="*60)
    print()


# --------------------------------------------------
# Endpoint
# --------------------------------------------------

@app.get("/health", include_in_schema=False)
def health():
    return {"status": "ok"}


@app.get("/")
def health_check():
    """
    Health check endpoint.
    
    Returns system status and basic statistics.
    """
    return {
        "status": "healthy",
        "service": "E-Commerce Recommendation API",
        "version": "1.0.1",
        "mode": "Hybrid (LGB + MF with manual scoring)",
        "model_version": MODEL_VERSION,
        "stats": {
            "total_events": len(events_df) if events_df is not None else 0,
            "unique_users": int(events_df["visitorid"].nunique()) if events_df is not None else 0,
            "unique_items": int(events_df["itemid"].nunique()) if events_df is not None else 0,
            "date_range": {
                "start": str(events_df['timestamp'].min()) if events_df is not None else None,
                "end": str(events_df['timestamp'].max()) if events_df is not None else None
            }
        },
        "model": {
            "version": MODEL_VERSION,
            "features": len(artifact['features']) if artifact else 0,
            "mf_users": len(recommender.user_to_idx) if recommender and recommender.user_to_idx else 0,
            "mf_items": len(recommender.item_to_idx) if recommender and recommender.item_to_idx else 0,
            "hybrid_alpha": MF_ALPHA
        }
    }


@app.get("/recommend/{user_id}")
def recommend(user_id: int, n: int = 5):
    """
    Get personalized recommendations for a user.
    
    Args:
        user_id: Visitor ID to generate recommendations for
        n: Number of recommendations to return (default: 5, max: 50)
    
    Returns:
        JSON with user_id and list of recommendations (itemid + score + breakdown)
    
    Raises:
        HTTPException 404: User not found in the system
        HTTPException 400: Invalid parameters
        HTTPException 503: Service not ready
    """
    try:
        logger.info(f"Received recommendation request for user_id={user_id}, n={n}")
        
        # Check if service is ready
        if events_df is None or recommender is None:
            raise HTTPException(status_code=503, detail="Service not ready")

        # Validate parameters
        if n < 1 or n > 50:
            raise HTTPException(
                status_code=400, 
                detail="Parameter 'n' must be between 1 and 50"
            )
        
        # Check if user exists
        if user_id not in events_df["visitorid"].values:
            raise HTTPException(
                status_code=404, 
                detail=f"User {user_id} not found in the system"
            )

        # Check if user is warm or cold
        is_warm_user = user_id in recommender.user_to_idx if recommender.user_to_idx else False
        logger.info(f"User {user_id} is {'WARM' if is_warm_user else 'COLD'}")

        # Generate candidate items
        logger.info(f"Generating candidate items for user {user_id}...")
        candidate_itemids = build_popular_items(
            events_df,
            user_id,
            max_candidates=200,
            item_to_idx=recommender.item_to_idx
        )
        logger.info(f"Generated {len(candidate_itemids)} candidates")

        if len(candidate_itemids) == 0:
            raise HTTPException(
                status_code=404, 
                detail=f"No candidate items found for user {user_id}"
            )

        # Build features for candidates
        logger.info(f"Building features for {len(candidate_itemids)} candidates...")
        feature_df = build_features(
            events_df,
            user_id,
            candidate_itemids
        )
        logger.info(f"Feature matrix shape: {feature_df.shape}")

        # Score candidates with hybrid model
        logger.info(f"Scoring candidates with hybrid model...")
        scored = recommender.score_candidates(feature_df, user_id=user_id)
        logger.info(f"Scored {len(scored)} items successfully")
        
        # Get top N recommendations
        top_n_results = (
            scored
            .sort_values("score", ascending=False)
            .head(n)[["itemid", "score", "lgb_score", "mf_score"]]
        )

        logger.info(f"Returning top {n} recommendations")
        return {
            "user_id": user_id,
            "user_type": "warm" if is_warm_user else "cold",
            "recommendations": top_n_results.to_dict(orient="records"),
            "total_candidates_scored": len(candidate_itemids)
        }
    
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        # Log the full traceback for debugging
        logger.error(f"Error generating recommendations for user {user_id}:")
        logger.error(f"Error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        
        # Return a 500 error with the actual error message
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.get("/user/{user_id}/stats")
def user_stats(user_id: int):
    """
    Get statistics about a user's behavior.
    
    Args:
        user_id: Visitor ID
    
    Returns:
        JSON with user statistics including warm/cold status
    """
    try:
        # Check if service is ready
        if events_df is None or recommender is None:
            raise HTTPException(status_code=503, detail="Service not ready")

        # Check if user exists
        if user_id not in events_df["visitorid"].values:
            raise HTTPException(
                status_code=404, 
                detail=f"User {user_id} not found in the system"
            )
        
        user_events = events_df[events_df["visitorid"] == user_id]
        
        # Check if user is in MF model (warm user)
        is_warm_user = user_id in recommender.user_to_idx if recommender.user_to_idx else False
        
        return {
            "user_id": user_id,
            "user_type": "warm" if is_warm_user else "cold",
            "total_events": int(len(user_events)),
            "purchases": int(user_events["label"].sum()),
            "conversion_rate": float(user_events["label"].mean()),
            "unique_items_viewed": int(user_events["itemid"].nunique()),
            "unique_categories_viewed": int(user_events["category_id"].nunique()),
            "first_seen": str(user_events["timestamp"].min()),
            "last_seen": str(user_events["timestamp"].max())
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting stats for user {user_id}:")
        logger.error(f"Error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


# --------------------------------------------------
# Main (for local testing with uvicorn)
# --------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)