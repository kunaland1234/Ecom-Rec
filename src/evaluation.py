"""
Model evaluation script with comprehensive metrics and visualizations.

Evaluates the trained model on:
- Classification metrics (ROC-AUC, precision, recall)
- Ranking metrics (precision@k, recall@k)
- Per-user performance analysis

Generates visualizations:
- ROC Curve
- Precision-Recall Curve
- Feature Importance
- Score Distribution
- Precision@K plot
"""

from src.preprocess import load_and_clean_events, load_item_categories
from src.build_feature import build_features
from src.candidate import build_popular_items
from src.recommender import Recommender
import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score, 
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix
)
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10


def create_output_dir():
    """Create directory for saving plots."""
    output_dir = Path("evaluation_results")
    output_dir.mkdir(exist_ok=True)
    return output_dir


def plot_roc_curve(y_test, y_pred, output_dir):
    """
    Plot ROC curve.
    
    Args:
        y_test: True labels
        y_pred: Predicted probabilities
        output_dir: Directory to save plot
    """
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='#2E86AB', linewidth=2.5, label=f'ROC Curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='#E63946', linestyle='--', linewidth=2, label='Random Classifier')
    
    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold', pad=20)
    plt.legend(loc='lower right', fontsize=11, frameon=True, shadow=True)
    plt.grid(True, alpha=0.3)
    
    # Add text box with key metrics
    textstr = f'AUC: {auc:.4f}\nBetter than random: {(auc-0.5)*100:.1f}%'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.6, 0.2, textstr, fontsize=11, bbox=props)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: roc_curve.png")


def plot_precision_recall_curve(y_test, y_pred, output_dir):
    """
    Plot Precision-Recall curve.
    
    Args:
        y_test: True labels
        y_pred: Predicted probabilities
        output_dir: Directory to save plot
    """
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
    ap = average_precision_score(y_test, y_pred)
    
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='#06A77D', linewidth=2.5, label=f'PR Curve (AP = {ap:.4f})')
    
    # Baseline (random classifier for imbalanced data)
    baseline = y_test.sum() / len(y_test)
    plt.axhline(y=baseline, color='#E63946', linestyle='--', linewidth=2, 
                label=f'Random Classifier (AP = {baseline:.4f})')
    
    plt.xlabel('Recall', fontsize=12, fontweight='bold')
    plt.ylabel('Precision', fontsize=12, fontweight='bold')
    plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold', pad=20)
    plt.legend(loc='upper right', fontsize=11, frameon=True, shadow=True)
    plt.grid(True, alpha=0.3)
    
    # Add text box
    textstr = f'Avg Precision: {ap:.4f}\nBaseline: {baseline:.4f}\nImprovement: {(ap/baseline):.2f}x'
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.5)
    plt.text(0.4, 0.95, textstr, fontsize=11, bbox=props, verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'precision_recall_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: precision_recall_curve.png")


def plot_feature_importance(model, features, output_dir, top_n=15):
    """
    Plot feature importance.
    
    Args:
        model: Trained model
        features: List of feature names
        output_dir: Directory to save plot
        top_n: Number of top features to show
    """
    feature_imp = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(top_n)
    
    plt.figure(figsize=(12, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(feature_imp)))
    
    bars = plt.barh(range(len(feature_imp)), feature_imp['importance'], color=colors)
    plt.yticks(range(len(feature_imp)), feature_imp['feature'])
    plt.xlabel('Importance Score', fontsize=12, fontweight='bold')
    plt.ylabel('Feature', fontsize=12, fontweight='bold')
    plt.title(f'Top {top_n} Most Important Features', fontsize=14, fontweight='bold', pad=20)
    plt.gca().invert_yaxis()
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, feature_imp['importance'])):
        plt.text(value, bar.get_y() + bar.get_height()/2, f'{value:.0f}', 
                va='center', ha='left', fontsize=9, fontweight='bold')
    
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: feature_importance.png")


def plot_score_distribution(y_test, y_pred, output_dir):
    """
    Plot distribution of prediction scores for positive and negative classes.
    
    Args:
        y_test: True labels
        y_pred: Predicted probabilities
        output_dir: Directory to save plot
    """
    plt.figure(figsize=(12, 6))
    
    # Separate scores by actual class
    positive_scores = y_pred[y_test == 1]
    negative_scores = y_pred[y_test == 0]
    
    # Plot distributions
    plt.hist(negative_scores, bins=50, alpha=0.6, color='#E63946', 
             label=f'Non-Purchases (n={len(negative_scores):,})', edgecolor='black')
    plt.hist(positive_scores, bins=50, alpha=0.6, color='#06A77D', 
             label=f'Purchases (n={len(positive_scores):,})', edgecolor='black')
    
    plt.xlabel('Predicted Probability', fontsize=12, fontweight='bold')
    plt.ylabel('Frequency (log scale)', fontsize=12, fontweight='bold')
    plt.title('Distribution of Model Predictions by Class', fontsize=14, fontweight='bold', pad=20)
    plt.legend(loc='upper right', fontsize=11, frameon=True, shadow=True)
    plt.yscale('log')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add median lines
    plt.axvline(np.median(positive_scores), color='#06A77D', linestyle='--', linewidth=2, 
                label=f'Median (Purchases): {np.median(positive_scores):.4f}')
    plt.axvline(np.median(negative_scores), color='#E63946', linestyle='--', linewidth=2,
                label=f'Median (Non-Purchases): {np.median(negative_scores):.4f}')
    
    plt.legend(loc='upper right', fontsize=10, frameon=True, shadow=True)
    plt.tight_layout()
    plt.savefig(output_dir / 'score_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: score_distribution.png")


def plot_precision_at_k(precisions, recalls, output_dir):
    """
    Plot Precision@K and Recall@K metrics.
    
    Args:
        precisions: Dictionary of {k: precision_values}
        recalls: Dictionary of {k: recall_values}
        output_dir: Directory to save plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    k_values = sorted(precisions.keys())
    precision_means = [np.mean(precisions[k]) for k in k_values]
    recall_means = [np.mean(recalls[k]) for k in k_values]
    
    # Precision@K plot
    ax1.plot(k_values, precision_means, marker='o', markersize=10, linewidth=2.5, 
             color='#2E86AB', label='Precision@K')
    ax1.fill_between(k_values, precision_means, alpha=0.3, color='#2E86AB')
    ax1.set_xlabel('K (Number of Recommendations)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax1.set_title('Precision@K', fontsize=14, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(k_values)
    
    # Add value labels
    for k, p in zip(k_values, precision_means):
        ax1.text(k, p + 0.01, f'{p:.3f}', ha='center', fontsize=10, fontweight='bold')
    
    # Recall@K plot
    ax2.plot(k_values, recall_means, marker='s', markersize=10, linewidth=2.5, 
             color='#06A77D', label='Recall@K')
    ax2.fill_between(k_values, recall_means, alpha=0.3, color='#06A77D')
    ax2.set_xlabel('K (Number of Recommendations)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Recall', fontsize=12, fontweight='bold')
    ax2.set_title('Recall@K', fontsize=14, fontweight='bold', pad=15)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(k_values)
    
    # Add value labels
    for k, r in zip(k_values, recall_means):
        ax2.text(k, r + 0.01, f'{r:.3f}', ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'precision_recall_at_k.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: precision_recall_at_k.png")


def plot_confusion_matrix_heatmap(y_test, y_pred, threshold=0.5, output_dir=None):
    """
    Plot confusion matrix heatmap.
    
    Args:
        y_test: True labels
        y_pred: Predicted probabilities
        threshold: Classification threshold
        output_dir: Directory to save plot
    """
    y_pred_binary = (y_pred >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred_binary)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['Non-Purchase', 'Purchase'],
                yticklabels=['Non-Purchase', 'Purchase'],
                annot_kws={'size': 14, 'weight': 'bold'})
    
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.title(f'Confusion Matrix (threshold={threshold})', fontsize=14, fontweight='bold', pad=20)
    
    # Calculate metrics
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    textstr = f'Accuracy: {accuracy:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(1.5, -0.15, textstr, fontsize=11, bbox=props, transform=plt.gca().transAxes)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: confusion_matrix.png")


def create_metrics_summary_image(metrics_dict, output_dir):
    """
    Create a visual summary of key metrics.
    
    Args:
        metrics_dict: Dictionary of metric names and values
        output_dir: Directory to save plot
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    # Title
    fig.suptitle('📊 Model Performance Summary', fontsize=20, fontweight='bold', y=0.98)
    
    # Create text content
    text_content = []
    
    # Classification Metrics
    text_content.append(("Classification Metrics", 'bold', 16, '#2E86AB'))
    text_content.append((f"  ROC-AUC: {metrics_dict['roc_auc']:.4f}", 'normal', 14, 'black'))
    text_content.append((f"  Average Precision: {metrics_dict['avg_precision']:.4f}", 'normal', 14, 'black'))
    text_content.append(("", 'normal', 12, 'black'))
    
    # Prediction Stats
    text_content.append(("Prediction Statistics", 'bold', 16, '#06A77D'))
    text_content.append((f"  Min Score: {metrics_dict['pred_min']:.6f}", 'normal', 14, 'black'))
    text_content.append((f"  Max Score: {metrics_dict['pred_max']:.6f}", 'normal', 14, 'black'))
    text_content.append((f"  Mean Score: {metrics_dict['pred_mean']:.6f}", 'normal', 14, 'black'))
    text_content.append((f"  Median Score: {metrics_dict['pred_median']:.6f}", 'normal', 14, 'black'))
    text_content.append(("", 'normal', 12, 'black'))
    
    # Ranking Metrics
    text_content.append(("Ranking Metrics", 'bold', 16, '#E63946'))
    text_content.append((f"  Precision@5: {metrics_dict['precision_5']:.4f}", 'normal', 14, 'black'))
    text_content.append((f"  Recall@5: {metrics_dict['recall_5']:.4f}", 'normal', 14, 'black'))
    text_content.append((f"  Precision@10: {metrics_dict['precision_10']:.4f}", 'normal', 14, 'black'))
    text_content.append((f"  Recall@10: {metrics_dict['recall_10']:.4f}", 'normal', 14, 'black'))
    text_content.append((f"  Precision@20: {metrics_dict['precision_20']:.4f}", 'normal', 14, 'black'))
    text_content.append((f"  Recall@20: {metrics_dict['recall_20']:.4f}", 'normal', 14, 'black'))
    text_content.append(("", 'normal', 12, 'black'))
    
    # Dataset Info
    text_content.append(("Dataset Information", 'bold', 16, '#F77F00'))
    text_content.append((f"  Test Samples: {metrics_dict['test_samples']:,}", 'normal', 14, 'black'))
    text_content.append((f"  Positive Rate: {metrics_dict['positive_rate']:.2%}", 'normal', 14, 'black'))
    text_content.append((f"  Users Evaluated: {metrics_dict['users_evaluated']:,}", 'normal', 14, 'black'))
    
    # Draw text
    y_position = 0.85
    for text, weight, size, color in text_content:
        ax.text(0.1, y_position, text, fontsize=size, fontweight=weight, 
                color=color, transform=ax.transAxes, family='monospace')
        y_position -= 0.05
    
    # Add border
    rect = plt.Rectangle((0.05, 0.05), 0.9, 0.9, fill=False, edgecolor='black', 
                         linewidth=2, transform=ax.transAxes)
    ax.add_patch(rect)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: metrics_summary.png")


def evaluate_model():
    """
    Run comprehensive model evaluation with visualizations.
    """
    print("="*60)
    print("MODEL EVALUATION WITH VISUALIZATIONS")
    print("="*60)
    
    # Create output directory
    output_dir = create_output_dir()
    print(f"\n📁 Saving results to: {output_dir}/")
    
    # Load data
    print("\nLoading raw data...")
    cat_df = load_item_categories(
        "data/item_properties_part1.csv",
        "data/item_properties_part2.csv"
    )
    
    df_full = pd.read_csv("data/events.csv")
    df_full["timestamp"] = pd.to_datetime(df_full["timestamp"], unit="ms")
    df_full = df_full.sort_values("timestamp")
    
    print(f"Loaded {len(df_full):,} events")
    print(f"Date range: {df_full['timestamp'].min()} to {df_full['timestamp'].max()}")
    
    # Temporal split
    cutoff_time = df_full["timestamp"].quantile(0.8)
    print(f"Split at: {cutoff_time}")
    
    train_raw = df_full[df_full["timestamp"] <= cutoff_time]
    test_raw = df_full[df_full["timestamp"] > cutoff_time]
    
    print(f"Train: {len(train_raw):,}, Test: {len(test_raw):,}")
    
    # Preprocess
    train_raw.to_csv("temp_train.csv", index=False)
    test_raw.to_csv("temp_test.csv", index=False)
    
    print("\nPreprocessing training data...")
    train_df, train_stats = load_and_clean_events("temp_train.csv", cat_df, is_train=True)
    
    print("Preprocessing test data...")
    test_df = load_and_clean_events("temp_test.csv", cat_df, is_train=False, train_stats=train_stats)
    
    # Load model
    print("\nLoading model...")
    model_path = "models/model_v3.pkl"
    artifact = joblib.load(model_path)
    recommender = Recommender(model_path)
    
    FEATURES = artifact["features"]
    
    # Prepare test data
    X_test = test_df[FEATURES]
    y_test = test_df["label"]
    
    print("\n" + "="*60)
    print("CLASSIFICATION METRICS")
    print("="*60)
    
    # Predict
    y_pred = artifact["model"].predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    auc = roc_auc_score(y_test, y_pred)
    ap = average_precision_score(y_test, y_pred)
    
    print(f"\nROC-AUC: {auc:.4f}")
    print(f"Average Precision: {ap:.4f}")
    print(f"\nPrediction Statistics:")
    print(f"  Min: {y_pred.min():.6f}")
    print(f"  Max: {y_pred.max():.6f}")
    print(f"  Mean: {y_pred.mean():.6f}")
    print(f"  Median: {np.median(y_pred):.6f}")
    
    print("\n" + "="*60)
    print("RANKING METRICS")
    print("="*60)
    
    # Evaluate ranking performance per user
    test_users = test_df[test_df["label"] == 1]["visitorid"].unique()
    print(f"\nEvaluating {len(test_users):,} users with purchases...")
    
    precisions = {5: [], 10: [], 20: []}
    recalls = {5: [], 10: [], 20: []}
    
    for user_id in test_users[:1000]:  # Evaluate on first 1000 users for speed
        # Get user's actual purchases in test set
        user_test_purchases = set(
            test_df[(test_df["visitorid"] == user_id) & (test_df["label"] == 1)]["itemid"]
        )
        
        if len(user_test_purchases) == 0:
            continue
        
        # Get candidates and score them
        try:
            candidates = build_popular_items(train_df, user_id, max_candidates=100)
            if len(candidates) == 0:
                continue
            
            feature_df = build_features(train_df, user_id, candidates)
            scored = recommender.score_candidates(feature_df)
            
            # Get top recommendations at different K values
            for k in [5, 10, 20]:
                top_k = set(recommender.top_n(scored, n=k)["itemid"])
                
                if len(top_k) > 0:
                    p = len(top_k & user_test_purchases) / len(top_k)
                    r = len(top_k & user_test_purchases) / len(user_test_purchases)
                    precisions[k].append(p)
                    recalls[k].append(r)
                
        except Exception as e:
            continue
    
    # Print ranking metrics
    print(f"\nRanking Metrics (evaluated on {len(precisions[5])} users):")
    for k in [5, 10, 20]:
        print(f"Precision@{k}: {np.mean(precisions[k]):.4f}")
        print(f"Recall@{k}:    {np.mean(recalls[k]):.4f}")
    
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    print()
    
    # Generate all plots
    plot_roc_curve(y_test, y_pred, output_dir)
    plot_precision_recall_curve(y_test, y_pred, output_dir)
    plot_feature_importance(artifact["model"], FEATURES, output_dir)
    plot_score_distribution(y_test, y_pred, output_dir)
    plot_precision_at_k(precisions, recalls, output_dir)
    plot_confusion_matrix_heatmap(y_test, y_pred, threshold=0.5, output_dir=output_dir)
    
    # Create metrics summary
    metrics_dict = {
        'roc_auc': auc,
        'avg_precision': ap,
        'pred_min': y_pred.min(),
        'pred_max': y_pred.max(),
        'pred_mean': y_pred.mean(),
        'pred_median': np.median(y_pred),
        'precision_5': np.mean(precisions[5]),
        'recall_5': np.mean(recalls[5]),
        'precision_10': np.mean(precisions[10]),
        'recall_10': np.mean(recalls[10]),
        'precision_20': np.mean(precisions[20]),
        'recall_20': np.mean(recalls[20]),
        'test_samples': len(y_test),
        'positive_rate': y_test.mean(),
        'users_evaluated': len(precisions[5])
    }
    
    create_metrics_summary_image(metrics_dict, output_dir)
    
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE")
    print("="*60)
    
    feature_imp = pd.DataFrame({
        'feature': FEATURES,
        'importance': artifact["model"].feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Features:")
    for idx, row in feature_imp.head(10).iterrows():
        print(f"  {row['feature']:30s} {row['importance']:10.1f}")
    
    print("\n" + "="*60)
    print("✅ EVALUATION COMPLETE")
    print("="*60)
    print(f"\n📊 All visualizations saved to: {output_dir}/")
    print("\nGenerated files:")
    print("  • roc_curve.png")
    print("  • precision_recall_curve.png")
    print("  • feature_importance.png")
    print("  • score_distribution.png")
    print("  • precision_recall_at_k.png")
    print("  • confusion_matrix.png")
    print("  • metrics_summary.png")


if __name__ == "__main__":
    evaluate_model()