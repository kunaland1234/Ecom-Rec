"""
STEP 1: Create train/test splits from raw data
Run this FIRST!
"""

import pandas as pd
from pathlib import Path
from src.preprocess import load_and_clean_events, load_item_categories


def main():
    print("\n" + "="*70)
    print("PREPROCESSING: Creating Train/Test Splits")
    print("="*70)
    
    # Load item categories
    print("\n📂 Loading item categories...")
    cat_df = load_item_categories(
        "data/item_properties_part1.csv",
        "data/item_properties_part2.csv"
    )
    print(f"  ✅ {len(cat_df):,} items with categories")
    
    # Load raw events
    print("\n📂 Loading raw events...")
    df_full = pd.read_csv("data/events.csv")
    df_full["timestamp"] = pd.to_datetime(df_full["timestamp"], unit="ms")
    df_full = df_full.sort_values("timestamp")
    print(f"  ✅ {len(df_full):,} events")
    print(f"  📅 {df_full['timestamp'].min()} to {df_full['timestamp'].max()}")
    
    # Split train/test (80/20 by time)
    print("\n✂️ Splitting train/test (80/20 by time)...")
    cutoff = df_full["timestamp"].quantile(0.8)
    train_raw = df_full[df_full["timestamp"] <= cutoff]
    test_raw = df_full[df_full["timestamp"] > cutoff]
    print(f"  ✅ Train: {len(train_raw):,} events")
    print(f"  ✅ Test:  {len(test_raw):,} events")
    
    # Save temp files
    Path("data/temp").mkdir(exist_ok=True)
    train_raw.to_csv("data/temp/train_raw.csv", index=False)
    test_raw.to_csv("data/temp/test_raw.csv", index=False)
    
    # Preprocess train (compute statistics)
    print("\n🔧 Preprocessing TRAIN data...")
    train_df, train_stats = load_and_clean_events(
        "data/temp/train_raw.csv", cat_df, is_train=True
    )
    print(f"  ✅ {len(train_df):,} events with {len(train_df.columns)} features")
    
    # Preprocess test (use train statistics)
    print("\n🔧 Preprocessing TEST data...")
    test_df = load_and_clean_events(
        "data/temp/test_raw.csv", cat_df, is_train=False, train_stats=train_stats
    )
    print(f"  ✅ {len(test_df):,} events")
    
    # Save preprocessed data
    print("\n💾 Saving preprocessed files...")
    train_df.to_csv("data/preprocessed_train.csv", index=False)
    test_df.to_csv("data/preprocessed_test.csv", index=False)
    print(f"  ✅ data/preprocessed_train.csv")
    print(f"  ✅ data/preprocessed_test.csv")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\nTRAIN: {len(train_df):,} events")
    print(f"  Users: {train_df['visitorid'].nunique():,}")
    print(f"  Items: {train_df['itemid'].nunique():,}")
    print(f"  Purchases: {train_df['label'].sum():,} ({train_df['label'].mean()*100:.2f}%)")
    
    print(f"\nTEST: {len(test_df):,} events")
    print(f"  Users: {test_df['visitorid'].nunique():,}")
    print(f"  Items: {test_df['itemid'].nunique():,}")
    print(f"  Purchases: {test_df['label'].sum():,} ({test_df['label'].mean()*100:.2f}%)")
    

if __name__ == "__main__":
    main()