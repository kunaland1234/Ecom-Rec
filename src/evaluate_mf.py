"""
STEP 3: Evaluate Matrix Factorization Model
Tests ranking quality, coverage, diversity, and cold start handling
Creates visualizations similar to LightGBM evaluation
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, auc
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10


def evaluate_mf():
    print("\n" + "="*70)
    print("MATRIX FACTORIZATION EVALUATION")
    print("="*70)
    
    # Create output directory for plots
    output_dir = Path("evaluation_results_MF")
    output_dir.mkdir(exist_ok=True)
    print(f"\n📁 Saving results to: {output_dir}/")
    
    # Load model
    print("\n📂 Loading MF model...")
    mf_data = joblib.load("models/mf_model.pkl")
    model = mf_data['model']
    user_to_idx = mf_data['user_to_idx']
    item_to_idx = mf_data['item_to_idx']
    idx_to_item = mf_data['idx_to_item']
    print(f"  ✅ Model loaded ({len(user_to_idx):,} users, {len(item_to_idx):,} items)")
    
    # Load test data
    print("\n📂 Loading test data...")
    test_df = pd.read_csv("data/preprocessed_test.csv")
    print(f"  ✅ {len(test_df):,} test events")
    
    # Load train data (needed for filtering seen items)
    print("\n📂 Loading train data...")
    train_df = pd.read_csv("data/preprocessed_train.csv")
    print(f"  ✅ {len(train_df):,} train events")
    
    # ========================================================================
    # 1. RANKING METRICS (Precision@K, Recall@K) + ROC/PR Curves
    # ========================================================================
    print("\n" + "="*70)
    print("1. RANKING METRICS & PREDICTION QUALITY")
    print("="*70)
    
    # Get users who made purchases in test
    test_purchases = test_df[test_df['label'] == 1]
    test_users = test_purchases['visitorid'].unique()
    print(f"\nTest users with purchases: {len(test_users):,}")
    
    k_values = [5, 10, 20]
    results = {k: {'precisions': [], 'recalls': [], 'hits': 0} for k in k_values}
    
    # For ROC and PR curves - collect all predictions
    all_scores = []
    all_labels = []
    
    evaluated = 0
    cold_start = 0
    
    print("\nEvaluating users...")
    for i, user_id in enumerate(test_users):
        if i % 100 == 0:
            print(f"  Progress: {i}/{len(test_users)}", end='\r')
        
        # Skip cold start users
        if user_id not in user_to_idx:
            cold_start += 1
            continue
        
        # Get user's test purchases
        actual_purchases = set(
            test_purchases[test_purchases['visitorid'] == user_id]['itemid']
        )
        
        if len(actual_purchases) == 0:
            continue
        
        # Get user's test events (for score prediction)
        user_test_events = test_df[test_df['visitorid'] == user_id]
        
        # Get recommendations
        user_idx = user_to_idx[user_id]
        
        try:
            # Get top 50 recommendations (don't exclude seen - we'll filter manually)
            rec_indices = model.topN(user=user_idx, n=50, exclude_seen=False)
            recommendations = [idx_to_item[idx] for idx in rec_indices]
            
            # Filter out items the user already purchased IN TRAINING
            train_purchases_user = set(
                train_df[(train_df['visitorid'] == user_id) & (train_df['label'] == 1)]['itemid']
            )
            recommendations = [item for item in recommendations if item not in train_purchases_user]
            
            if len(recommendations) == 0:
                continue
            
            # Get scores for all test events of this user
            for _, event in user_test_events.iterrows():
                item_id = event['itemid']
                if item_id in item_to_idx:
                    item_idx = item_to_idx[item_id]
                    try:
                        score = model.predict(user=user_idx, item=item_idx)
                        all_scores.append(float(score))
                        all_labels.append(event['label'])
                    except:
                        pass
            
            # Evaluate at each K
            for k in k_values:
                top_k = set(recommendations[:k])
                hits = len(top_k & actual_purchases)
                
                if len(top_k) > 0:
                    precision = hits / len(top_k)
                    recall = hits / len(actual_purchases)
                    
                    results[k]['precisions'].append(precision)
                    results[k]['recalls'].append(recall)
                    if hits > 0:
                        results[k]['hits'] += 1
            
            evaluated += 1
            
        except Exception as e:
            continue
    
    print(f"\n\n📊 Ranking Results:")
    print(f"  Evaluated: {evaluated:,} users")
    print(f"  Cold start: {cold_start:,} users")
    
    print(f"\n{'K':<5} {'Precision':<12} {'Recall':<12} {'Hit Rate':<12}")
    print("-" * 50)
    
    for k in k_values:
        if results[k]['precisions']:
            precision = np.mean(results[k]['precisions'])
            recall = np.mean(results[k]['recalls'])
            hit_rate = results[k]['hits'] / evaluated
            print(f"{k:<5} {precision:<12.4f} {recall:<12.4f} {hit_rate:<12.4f}")
    
    # Plot Precision-Recall at K
    print("\n📊 Creating Precision-Recall at K plot...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    k_list = k_values
    precisions_at_k = [np.mean(results[k]['precisions']) for k in k_values]
    recalls_at_k = [np.mean(results[k]['recalls']) for k in k_values]
    
    x = np.arange(len(k_list))
    width = 0.35
    
    ax.bar(x - width/2, precisions_at_k, width, label='Precision@K', alpha=0.8)
    ax.bar(x + width/2, recalls_at_k, width, label='Recall@K', alpha=0.8)
    
    ax.set_xlabel('K (Top-K Recommendations)')
    ax.set_ylabel('Score')
    ax.set_title('Matrix Factorization: Precision & Recall at K')
    ax.set_xticks(x)
    ax.set_xticklabels([f'K={k}' for k in k_list])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'precision_recall_at_k.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: {output_dir}/precision_recall_at_k.png")
    
    # ROC Curve
    if len(all_scores) > 0 and len(set(all_labels)) > 1:
        print("\n📊 Creating ROC curve...")
        fpr, tpr, _ = roc_curve(all_labels, all_scores)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'MF ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Matrix Factorization: ROC Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'roc_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✅ Saved: {output_dir}/roc_curve.png")
        print(f"  ✅ ROC AUC: {roc_auc:.4f}")
        
        # Precision-Recall Curve
        print("\n📊 Creating Precision-Recall curve...")
        precision_curve, recall_curve, _ = precision_recall_curve(all_labels, all_scores)
        pr_auc = auc(recall_curve, precision_curve)
        
        plt.figure(figsize=(10, 8))
        plt.plot(recall_curve, precision_curve, color='blue', lw=2,
                label=f'MF PR curve (AUC = {pr_auc:.4f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Matrix Factorization: Precision-Recall Curve')
        plt.legend(loc="upper right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'precision_recall_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✅ Saved: {output_dir}/precision_recall_curve.png")
        print(f"  ✅ PR AUC: {pr_auc:.4f}")
    
    # ========================================================================
    # 2. COVERAGE & DIVERSITY
    # ========================================================================
    print("\n" + "="*70)
    print("2. CATALOG COVERAGE & DIVERSITY")
    print("="*70)
    
    total_items = len(item_to_idx)
    recommended_items = set()
    item_recommendation_counts = {}
    
    sample_users = [u for u in test_users[:1000] if u in user_to_idx]
    
    print(f"\nSampling {len(sample_users)} users for coverage analysis...")
    for user_id in sample_users:
        user_idx = user_to_idx[user_id]
        try:
            rec_indices = model.topN(user=user_idx, n=50, exclude_seen=False)
            items = [idx_to_item[idx] for idx in rec_indices]
            
            # Filter out training purchases
            train_purchases_user = set(
                train_df[(train_df['visitorid'] == user_id) & (train_df['label'] == 1)]['itemid']
            )
            items = [item for item in items if item not in train_purchases_user]
            
            recommended_items.update(items)
            
            # Count how often each item is recommended
            for item in items:
                item_recommendation_counts[item] = item_recommendation_counts.get(item, 0) + 1
        except:
            continue
    
    coverage = len(recommended_items) / total_items
    
    print(f"\n📊 Coverage Results:")
    print(f"  Total catalog items: {total_items:,}")
    print(f"  Items recommended: {len(recommended_items):,}")
    print(f"  Coverage: {coverage:.4f} ({coverage*100:.2f}%)")
    
    # Plot item recommendation distribution
    print("\n📊 Creating item recommendation distribution plot...")
    if item_recommendation_counts:
        counts = list(item_recommendation_counts.values())
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Histogram of recommendation counts
        ax1.hist(counts, bins=50, edgecolor='black', alpha=0.7)
        ax1.set_xlabel('Number of times recommended')
        ax1.set_ylabel('Number of items')
        ax1.set_title('Item Recommendation Frequency Distribution')
        ax1.grid(True, alpha=0.3)
        
        # Top 20 most recommended items
        top_items = sorted(item_recommendation_counts.items(), key=lambda x: x[1], reverse=True)[:20]
        items_list = [f"Item {x[0]}" for x in top_items]
        counts_list = [x[1] for x in top_items]
        
        ax2.barh(range(len(items_list)), counts_list, alpha=0.7)
        ax2.set_yticks(range(len(items_list)))
        ax2.set_yticklabels(items_list, fontsize=8)
        ax2.set_xlabel('Recommendation Count')
        ax2.set_title('Top 20 Most Recommended Items')
        ax2.invert_yaxis()
        ax2.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'item_recommendation_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✅ Saved: {output_dir}/item_recommendation_distribution.png")
    
    # Diversity Analysis
    print("\n📊 Calculating recommendation diversity...")
    user_recs = []
    sample_users_div = [u for u in test_users[:100] if u in user_to_idx]
    
    for user_id in sample_users_div:
        user_idx = user_to_idx[user_id]
        try:
            rec_indices = model.topN(user=user_idx, n=10, exclude_seen=False)
            recs_items = [idx_to_item[idx] for idx in rec_indices]
            
            # Filter out training purchases
            train_purchases_user = set(
                train_df[(train_df['visitorid'] == user_id) & (train_df['label'] == 1)]['itemid']
            )
            recs_items = [item for item in recs_items if item not in train_purchases_user]
            
            recs = set(recs_items)
            if recs:
                user_recs.append(recs)
        except:
            continue
    
    # Calculate pairwise Jaccard similarity
    diversity = 0
    if len(user_recs) > 1:
        similarities = []
        for i in range(min(50, len(user_recs))):
            for j in range(i+1, min(50, len(user_recs))):
                intersection = len(user_recs[i] & user_recs[j])
                union = len(user_recs[i] | user_recs[j])
                if union > 0:
                    similarity = intersection / union
                    similarities.append(similarity)
        
        if similarities:
            avg_sim = np.mean(similarities)
            diversity = 1 - avg_sim
            
            print(f"\n📊 Diversity Results:")
            print(f"  Avg Jaccard similarity: {avg_sim:.4f}")
            print(f"  Diversity score: {diversity:.4f}")
            
            # Plot diversity distribution
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(similarities, bins=30, edgecolor='black', alpha=0.7)
            ax.axvline(avg_sim, color='red', linestyle='--', linewidth=2, 
                      label=f'Mean Similarity: {avg_sim:.3f}')
            ax.set_xlabel('Jaccard Similarity between User Recommendations')
            ax.set_ylabel('Frequency')
            ax.set_title(f'Recommendation Diversity (Diversity Score: {diversity:.3f})')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'recommendation_diversity.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  ✅ Saved: {output_dir}/recommendation_diversity.png")
    
    # ========================================================================
    # 3. COLD START ANALYSIS
    # ========================================================================
    print("\n" + "="*70)
    print("3. COLD START ANALYSIS")
    print("="*70)
    
    # User cold start
    train_users = set(train_df['visitorid'].unique())
    test_users_all = set(test_df['visitorid'].unique())
    cold_users = test_users_all - train_users
    
    user_cold_rate = len(cold_users) / len(test_users_all)
    
    print(f"\n👤 User Cold Start:")
    print(f"  Total test users: {len(test_users_all):,}")
    print(f"  Cold start users: {len(cold_users):,}")
    print(f"  Cold start rate: {user_cold_rate:.4f} ({user_cold_rate*100:.2f}%)")
    
    # Item cold start
    train_items = set(train_df['itemid'].unique())
    test_items_all = set(test_df['itemid'].unique())
    cold_items = test_items_all - train_items
    
    item_cold_rate = len(cold_items) / len(test_items_all)
    
    print(f"\n📦 Item Cold Start:")
    print(f"  Total test items: {len(test_items_all):,}")
    print(f"  Cold start items: {len(cold_items):,}")
    print(f"  Cold start rate: {item_cold_rate:.4f} ({item_cold_rate*100:.2f}%)")
    
    # Plot cold start analysis
    print("\n📊 Creating cold start visualization...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # User cold start pie chart
    user_labels = ['Known Users', 'Cold Start Users']
    user_sizes = [len(test_users_all) - len(cold_users), len(cold_users)]
    user_colors = ['#2ecc71', '#e74c3c']
    
    ax1.pie(user_sizes, labels=user_labels, autopct='%1.1f%%', startangle=90,
           colors=user_colors, textprops={'fontsize': 11})
    ax1.set_title(f'User Cold Start\n({len(cold_users):,} / {len(test_users_all):,} users are new)')
    
    # Item cold start pie chart
    item_labels = ['Known Items', 'Cold Start Items']
    item_sizes = [len(test_items_all) - len(cold_items), len(cold_items)]
    item_colors = ['#3498db', '#e67e22']
    
    ax2.pie(item_sizes, labels=item_labels, autopct='%1.1f%%', startangle=90,
           colors=item_colors, textprops={'fontsize': 11})
    ax2.set_title(f'Item Cold Start\n({len(cold_items):,} / {len(test_items_all):,} items are new)')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'cold_start_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: {output_dir}/cold_start_analysis.png")
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    
    summary_metrics = {
        'Precision@5': np.mean(results[5]['precisions']) if results[5]['precisions'] else 0,
        'Precision@10': np.mean(results[10]['precisions']) if results[10]['precisions'] else 0,
        'Precision@20': np.mean(results[20]['precisions']) if results[20]['precisions'] else 0,
        'Recall@5': np.mean(results[5]['recalls']) if results[5]['recalls'] else 0,
        'Recall@10': np.mean(results[10]['recalls']) if results[10]['recalls'] else 0,
        'Recall@20': np.mean(results[20]['recalls']) if results[20]['recalls'] else 0,
        'Coverage': coverage,
        'Diversity': diversity if 'diversity' in locals() else 0,
        'User Cold Start %': user_cold_rate * 100,
        'Item Cold Start %': item_cold_rate * 100
    }
    
    print(f"\n🎯 Ranking Metrics:")
    print(f"  Precision@10: {summary_metrics['Precision@10']:.4f}")
    print(f"  Recall@10: {summary_metrics['Recall@10']:.4f}")
    print(f"\n🎨 Quality Metrics:")
    print(f"  Coverage: {summary_metrics['Coverage']:.4f}")
    print(f"  Diversity: {summary_metrics['Diversity']:.4f}")
    print(f"\n❄️  Cold Start:")
    print(f"  User: {summary_metrics['User Cold Start %']:.2f}%")
    print(f"  Item: {summary_metrics['Item Cold Start %']:.2f}%")
    
    # Save summary metrics
    summary_df = pd.DataFrame([summary_metrics])
    summary_df.to_csv(output_dir / 'evaluation_summary.csv', index=False)
    print(f"\n💾 Saved summary: {output_dir}/evaluation_summary.csv")
    
    # Create summary visualization
    print("\n📊 Creating summary metrics visualization...")
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Precision@K
    k_vals = [5, 10, 20]
    precisions = [summary_metrics[f'Precision@{k}'] for k in k_vals]
    ax1.bar([f'@{k}' for k in k_vals], precisions, color='steelblue', alpha=0.7)
    ax1.set_ylabel('Precision')
    ax1.set_title('Precision at Different K Values')
    ax1.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(precisions):
        ax1.text(i, v + 0.005, f'{v:.3f}', ha='center', va='bottom')
    
    # Recall@K
    recalls = [summary_metrics[f'Recall@{k}'] for k in k_vals]
    ax2.bar([f'@{k}' for k in k_vals], recalls, color='coral', alpha=0.7)
    ax2.set_ylabel('Recall')
    ax2.set_title('Recall at Different K Values')
    ax2.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(recalls):
        ax2.text(i, v + 0.005, f'{v:.3f}', ha='center', va='bottom')
    
    # Coverage & Diversity
    quality_metrics = ['Coverage', 'Diversity']
    quality_values = [summary_metrics['Coverage'], summary_metrics['Diversity']]
    colors = ['#3498db', '#2ecc71']
    bars = ax3.bar(quality_metrics, quality_values, color=colors, alpha=0.7)
    ax3.set_ylabel('Score')
    ax3.set_title('Coverage & Diversity Scores')
    ax3.set_ylim([0, 1])
    ax3.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, quality_values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{val:.3f}', ha='center', va='bottom')
    
    # Cold Start Rates
    cold_start_metrics = ['User\nCold Start', 'Item\nCold Start']
    cold_start_values = [summary_metrics['User Cold Start %'], summary_metrics['Item Cold Start %']]
    bars = ax4.bar(cold_start_metrics, cold_start_values, color=['#e74c3c', '#e67e22'], alpha=0.7)
    ax4.set_ylabel('Percentage (%)')
    ax4.set_title('Cold Start Rates')
    ax4.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, cold_start_values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{val:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'evaluation_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: {output_dir}/evaluation_summary.png")
    
    print("\n" + "="*70)
    print("✅ EVALUATION COMPLETE")
    print("="*70)
    print(f"\n📁 All results saved to: {output_dir}/")
    print("\nGenerated plots:")
    print("  1. precision_recall_at_k.png")
    print("  2. roc_curve.png")
    print("  3. precision_recall_curve.png")
    print("  4. item_recommendation_distribution.png")
    print("  5. recommendation_diversity.png")
    print("  6. cold_start_analysis.png")
    print("  7. evaluation_summary.png")
    print("  8. evaluation_summary.csv")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    import os
    if not os.path.exists("models/mf_model.pkl"):
        print("\n❌ ERROR: MF model not found!")
        print("Run this first: python train_mf.py\n")
    else:
        evaluate_mf()