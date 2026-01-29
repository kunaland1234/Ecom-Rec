"""
Model training pipeline for e-commerce recommendation system.

This script:
1. Loads and preprocesses event data
2. Splits into train/test sets (80/20 temporal split)
3. Trains a LightGBM classifier
4. Evaluates on test set
5. Saves model artifact
"""

from src.preprocess import load_and_clean_events, load_item_categories
import joblib
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
import os

path = 'data/events.csv'


def train():
    """
    Main training pipeline.
    
    Returns:
        Tuple of (trained_model, test_features_with_ids, test_labels)
    """
    # Load item category mappings
    cat_df = load_item_categories(
        "data/item_properties_part1.csv",
        "data/item_properties_part2.csv"
    )

    print("="*60)
    print("LOADING AND SPLITTING DATA")
    print("="*60)
    
    # Load raw events
    df_full = pd.read_csv(path)
    df_full["timestamp"] = pd.to_datetime(df_full["timestamp"], unit="ms")
    df_full = df_full.sort_values("timestamp")
    
    print(f"\nLoaded {len(df_full)} events")
    print(f"Date range: {df_full['timestamp'].min()} to {df_full['timestamp'].max()}")
    
    # Temporal split at 80% mark (standard for time-series data)
    cutoff_time = df_full["timestamp"].quantile(0.8)
    print(f"Split at: {cutoff_time}")
    
    train_raw = df_full[df_full["timestamp"] <= cutoff_time]
    test_raw = df_full[df_full["timestamp"] > cutoff_time]
    
    print(f"Train: {len(train_raw)} events")
    print(f"Test: {len(test_raw)} events")
    
    # Save temporary files for preprocessing
    train_raw.to_csv("temp_train.csv", index=False)
    test_raw.to_csv("temp_test.csv", index=False)
    
    print("\n" + "="*60)
    print("FEATURE ENGINEERING")
    print("="*60)
    
    # Preprocess training data and compute statistics
    print("\nProcessing training data...")
    train_df, train_stats = load_and_clean_events(
        "temp_train.csv", 
        cat_df, 
        is_train=True
    )
    
    # Preprocess test data using training statistics
    print("Processing test data...")
    test_df = load_and_clean_events(
        "temp_test.csv", 
        cat_df, 
        is_train=False, 
        train_stats=train_stats
    )
    
    # Clean up temporary files
    os.remove("temp_train.csv")
    os.remove("temp_test.csv")
    
    print(f"\nTrain features: {train_df.shape}")
    print(f"Test features: {test_df.shape}")

    # Define feature set (simplified - removed problematic time-based features)
    FEATURES = [
        "category_id",
        "hour",
        "dayofweek",
        "is_weekend",
        "is_evening",
        "is_working_hours",
        "item_popularity",
        "item_conversion_rate",
        "recent_item_popularity",
        "item_popularity_ratio",
        "category_popularity",
        "category_conversion_rate",
        "user_event_count",
        "user_purchase_rate",
        "user_item_interaction_count",
        "user_category_affinity",
    ]

    TARGET = "label"

    # Prepare train/test sets
    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]
    X_test = test_df[FEATURES]
    y_test = test_df[TARGET]
    
    # Keep IDs for evaluation
    X_test_with_ids = X_test.copy()
    X_test_with_ids['itemid'] = test_df['itemid'].values
    X_test_with_ids['visitorid'] = test_df['visitorid'].values

    # Calculate class imbalance
    pos = y_train.sum()
    neg = len(y_train) - pos
    scale_pos_weight = neg / pos
    
    print("\n" + "="*60)
    print("TRAINING MODEL")
    print("="*60)
    print(f"\nClass distribution:")
    print(f"  Positive (purchases): {pos:,} ({pos/len(y_train)*100:.2f}%)")
    print(f"  Negative (non-purchases): {neg:,} ({neg/len(y_train)*100:.2f}%)")
    print(f"  Scale pos weight: {scale_pos_weight:.2f}")
    print(f"\nFeatures: {len(FEATURES)}")

    # Train LightGBM classifier
    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=7,
        num_leaves=63,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )

    print("\nTraining in progress...")
    model.fit(X_train, y_train)
    print("✅ Training complete!")
    
    print("\n" + "="*60)
    print("EVALUATION")
    print("="*60)
    
    # Predict on test set
    y_pred = model.predict_proba(X_test[FEATURES])[:, 1]
    
    # Calculate metrics
    auc = roc_auc_score(y_test, y_pred)
    ap = average_precision_score(y_test, y_pred)
    
    print(f"\nClassification Metrics:")
    print(f"  ROC-AUC: {auc:.4f}")
    print(f"  Average Precision: {ap:.4f}")
    
    print(f"\nPrediction Statistics:")
    print(f"  Min score: {y_pred.min():.6f}")
    print(f"  Max score: {y_pred.max():.6f}")
    print(f"  Mean score: {y_pred.mean():.6f}")
    print(f"  Median score: {pd.Series(y_pred).median():.6f}")

    # Feature importance analysis
    print(f"\nTop 10 Most Important Features:")
    feature_imp = pd.DataFrame({
        'feature': FEATURES,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for idx, row in feature_imp.head(10).iterrows():
        print(f"  {row['feature']:30s} {row['importance']:10.1f}")

    # Save model artifact
    print("\n" + "="*60)
    print("SAVING MODEL")
    print("="*60)
    
    artifact = {
        "model": model,
        "features": FEATURES,
        "train_stats": train_stats
    }

    os.makedirs("models", exist_ok=True)
    model_path = "models/model_v3.pkl"
    joblib.dump(artifact, model_path)
    
    print(f"\n✅ Model saved to {model_path}")
    print(f"   - Model type: {type(model).__name__}")
    print(f"   - Features: {len(FEATURES)}")
    print(f"   - Training samples: {len(X_train):,}")
    print(f"   - Test ROC-AUC: {auc:.4f}")
    print("\n" + "="*60)

    return model, X_test_with_ids, y_test


if __name__ == "__main__":
    train()