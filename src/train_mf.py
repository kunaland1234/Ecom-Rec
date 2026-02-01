"""
FIXED: Train Matrix Factorization Model
With proper error checking and better hyperparameters
"""

import pandas as pd
import numpy as np
import cmfrec
import joblib
from pathlib import Path


def train_mf():
    print("\n" + "="*70)
    print("MATRIX FACTORIZATION TRAINING (FIXED)")
    print("="*70)
    
    # Load data
    print("\n📂 Loading preprocessed data...")
    train_df = pd.read_csv("data/preprocessed_train.csv")
    print(f"  ✅ {len(train_df):,} events")
    
    # Check for NaN values
    print("\n🔍 Checking for NaN values...")
    nan_counts = train_df.isna().sum()
    if nan_counts.sum() > 0:
        print("  ⚠️  Found NaN values:")
        print(nan_counts[nan_counts > 0])
        print("  Filling NaN values with 0...")
        train_df = train_df.fillna(0)
    else:
        print("  ✅ No NaN values found")
    
    # Prepare interactions (weight purchases higher than views)
    print("\n🔧 Preparing user-item interactions...")
    train_df['weight'] = train_df['label'].apply(lambda x: 10.0 if x == 1 else 1.0)
    
    # Aggregate multiple interactions
    interactions = train_df.groupby(['visitorid', 'itemid'])['weight'].sum().reset_index()
    
    print(f"  ✅ {len(interactions):,} unique user-item pairs")
    print(f"  ✅ {interactions['visitorid'].nunique():,} users")
    print(f"  ✅ {interactions['itemid'].nunique():,} items")
    
    # Create ID mappings (MF needs sequential indices)
    print("\n🔢 Creating ID mappings...")
    users = interactions['visitorid'].unique()
    items = interactions['itemid'].unique()
    
    user_to_idx = {u: i for i, u in enumerate(users)}
    item_to_idx = {it: i for i, it in enumerate(items)}
    idx_to_user = {i: u for u, i in user_to_idx.items()}
    idx_to_item = {i: it for it, i in item_to_idx.items()}
    
    print(f"  ✅ Mapped {len(user_to_idx):,} users and {len(item_to_idx):,} items")
    
    # Map to indices
    interactions['user_idx'] = interactions['visitorid'].map(user_to_idx)
    interactions['item_idx'] = interactions['itemid'].map(item_to_idx)
    
    # Convert to integer
    interactions['user_idx'] = interactions['user_idx'].astype(int)
    interactions['item_idx'] = interactions['item_idx'].astype(int)
    
    # CMFrec needs DataFrame with specific column names: UserId, ItemId, Rating
    X_train = pd.DataFrame({
        'UserId': interactions['user_idx'].values,
        'ItemId': interactions['item_idx'].values,
        'Rating': interactions['weight'].values
    })
    
    print(f"\n  Training matrix shape: {X_train.shape}")
    print(f"  Rating range: {X_train['Rating'].min():.2f} to {X_train['Rating'].max():.2f}")
    
    # Prepare item features (optional but improves cold start)
    print("\n🎯 Preparing item features...")
    item_features = train_df[[
        'itemid', 
        'category_id',
        'item_popularity',
        'item_conversion_rate',
        'recent_item_popularity',
        'category_popularity',
        'category_conversion_rate'
    ]].drop_duplicates('itemid')
    
    item_features['item_idx'] = item_features['itemid'].map(item_to_idx)
    item_features = item_features.dropna(subset=['item_idx'])
    item_features = item_features.sort_values('item_idx')
    item_features['item_idx'] = item_features['item_idx'].astype(int)
    
    # Fill any NaN in features
    feature_cols = ['category_id', 'item_popularity', 'item_conversion_rate', 
                    'recent_item_popularity', 'category_popularity', 'category_conversion_rate']
    
    for col in feature_cols:
        if item_features[col].isna().any():
            print(f"  Filling NaN in {col}")
            item_features[col] = item_features[col].fillna(0)
    
    # Normalize features to prevent numerical issues
    print("\n📊 Normalizing features...")
    for col in ['item_popularity', 'recent_item_popularity', 'category_popularity']:
        if col in item_features.columns:
            max_val = item_features[col].max()
            if max_val > 0:
                item_features[col] = item_features[col] / max_val
    
    # CMFrec needs DataFrame with ItemId column + feature columns
    I_features = pd.DataFrame({
        'ItemId': item_features['item_idx'].values,
        'category_id': item_features['category_id'].values,
        'item_popularity': item_features['item_popularity'].values,
        'item_conversion_rate': item_features['item_conversion_rate'].values,
        'recent_item_popularity': item_features['recent_item_popularity'].values,
        'category_popularity': item_features['category_popularity'].values,
        'category_conversion_rate': item_features['category_conversion_rate'].values
    })
    
    print(f"  ✅ 6 features per item")
    print(f"  ✅ Feature DataFrame shape: {I_features.shape}")
    
    # Check for NaN in features
    if I_features.isna().any().any():
        print("  ⚠️  NaN found in item features, filling with 0")
        I_features = I_features.fillna(0)
    
    # Train model
    print("\n🎓 Training CMFrec model...")
    print(f"  - Latent factors: 50 (reduced from 100 for stability)")
    print(f"  - Regularization: 1.0 (increased for stability)")
    print(f"  - Iterations: 15")
    print(f"  - Item features: Yes (normalized)")
    print()
    
    model = cmfrec.CMF_implicit(
        k=50,                  # Reduced from 100 for stability
        lambda_=1.0,           # Increased from 0.01 for numerical stability
        niter=15,              # Reduced from 20
        use_cg=True,
        finalize_chol=True,
        random_state=42,
        verbose=True,
        nthreads=-1
    )
    
    try:
        model.fit(X=X_train, I=I_features)
        print("\n✅ Model trained successfully!")
        
        # Verify factors don't have NaN
        print("\n🔍 Verifying model factors...")
        user_nan = np.isnan(model.A_).sum()
        item_nan = np.isnan(model.B_).sum()
        
        print(f"  User factors NaN count: {user_nan} / {model.A_.size}")
        print(f"  Item factors NaN count: {item_nan} / {model.B_.size}")
        
        if user_nan > 0 or item_nan > 0:
            print("\n  ⚠️  WARNING: Model has NaN values!")
            print("  This may cause prediction issues.")
        else:
            print("  ✅ No NaN values in factors")
        
    except Exception as e:
        print(f"\n❌ ERROR during training: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Save model
    print("\n💾 Saving model...")
    Path("models").mkdir(exist_ok=True)
    
    model_data = {
        'model': model,
        'user_to_idx': user_to_idx,
        'item_to_idx': item_to_idx,
        'idx_to_user': idx_to_user,
        'idx_to_item': idx_to_item
    }
    
    joblib.dump(model_data, "models/mf_model.pkl")
    print(f"  ✅ models/mf_model.pkl saved")
    
    # Test a prediction
    print("\n🧪 Testing predictions...")
    try:
        # Get first user and item
        test_u = 0
        test_i = 0
        
        # Manual dot product
        score = np.dot(model.A_[test_u], model.B_[test_i])
        print(f"  Test prediction (user 0, item 0): {score}")
        
        if np.isnan(score):
            print("  ❌ Prediction is NaN!")
        else:
            print("  ✅ Prediction works!")
            
    except Exception as e:
        print(f"  ⚠️  Error testing prediction: {e}")
    


if __name__ == "__main__":
    import os
    if not os.path.exists("data/preprocessed_train.csv"):
        print("\n❌ ERROR: Preprocessed data not found!")
        print("Run this first: python run_preprocessing.py\n")
    else:
        train_mf()