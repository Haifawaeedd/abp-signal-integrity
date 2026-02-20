#!/usr/bin/env python3
"""
Complete ML Analysis Pipeline for ABP Signal Integrity
=======================================================
- Train/Test Split (70/30)
- Random Forest Classifier
- 5-Fold Cross-Validation
- Bootstrap Confidence Intervals (1000 iterations)
- Feature Importance Analysis
- ROC Curve and Confusion Matrix
- Low-FPR Threshold Analysis
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix, 
                             classification_report, precision_recall_curve,
                             average_precision_score)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("ML ANALYSIS PIPELINE - ABP SIGNAL INTEGRITY")
print("="*70)

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n[1/8] Loading data...")

df = pd.read_csv('features_vitaldb.csv')
print(f"  Total samples: {len(df)}")
print(f"  PASS: {(df['label'] == 'PASS').sum()}")
print(f"  FAIL: {(df['label'] == 'FAIL').sum()}")

# Prepare features
feature_cols = [
    'ABP_mean', 'ABP_std', 'ABP_min', 'ABP_max', 'ABP_range', 'ABP_IQR',
    'PP_mean', 'PP_std', 'PP_violation_ratio', 'MAP_violation_ratio',
    'SBP_violation_ratio', 'DBP_violation_ratio', 'PP_negative_ratio', 'PP_extreme_ratio',
    'negative_value_ratio', 'zero_value_ratio', 'flatline_ratio',
    'spike_ratio', 'saturation_ratio', 'missing_ratio'
]

# Use only columns that exist
existing_features = [c for c in feature_cols if c in df.columns]
print(f"  Features used: {len(existing_features)}")

X = df[existing_features].copy()
y = (df['label'] == 'FAIL').astype(int)

# Handle missing values
X = X.fillna(X.median())

# ============================================================================
# TRAIN/TEST SPLIT
# ============================================================================
print("\n[2/8] Train/Test Split (70/30)...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)

print(f"  Training set: {len(X_train)} samples")
print(f"  Test set: {len(X_test)} samples")
print(f"  Train FAIL rate: {100*y_train.mean():.1f}%")
print(f"  Test FAIL rate: {100*y_test.mean():.1f}%")

# ============================================================================
# RANDOM FOREST TRAINING
# ============================================================================
print("\n[3/8] Training Random Forest...")

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)
print("  ✓ Model trained")

# Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# ============================================================================
# CROSS-VALIDATION
# ============================================================================
print("\n[4/8] 5-Fold Cross-Validation...")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')

print(f"  CV ROC-AUC scores: {cv_scores}")
print(f"  Mean CV ROC-AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# ============================================================================
# BOOTSTRAP CONFIDENCE INTERVALS
# ============================================================================
print("\n[5/8] Bootstrap Confidence Intervals (1000 iterations)...")

B = 1000
bootstrap_aucs = []

np.random.seed(42)
for i in range(B):
    # Bootstrap sample
    idx = np.random.choice(len(X_test), size=len(X_test), replace=True)
    X_boot = X_test.iloc[idx]
    y_boot = y_test.iloc[idx]
    
    # Skip if only one class
    if len(np.unique(y_boot)) < 2:
        continue
    
    y_prob_boot = model.predict_proba(X_boot)[:, 1]
    auc = roc_auc_score(y_boot, y_prob_boot)
    bootstrap_aucs.append(auc)

bootstrap_aucs = np.array(bootstrap_aucs)
ci_lower = np.percentile(bootstrap_aucs, 2.5)
ci_upper = np.percentile(bootstrap_aucs, 97.5)

print(f"  Bootstrap samples: {len(bootstrap_aucs)}")
print(f"  Mean AUC: {bootstrap_aucs.mean():.3f}")
print(f"  95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")

# ============================================================================
# TEST SET METRICS
# ============================================================================
print("\n[6/8] Test Set Performance...")

test_auc = roc_auc_score(y_test, y_prob)
test_ap = average_precision_score(y_test, y_prob)

print(f"  ROC-AUC: {test_auc:.3f}")
print(f"  PR-AUC: {test_ap:.3f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"\n  Confusion Matrix:")
print(f"    TN: {tn}, FP: {fp}")
print(f"    FN: {fn}, TP: {tp}")

sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
npv = tn / (tn + fn) if (tn + fn) > 0 else 0

print(f"\n  Sensitivity (Recall): {100*sensitivity:.1f}%")
print(f"  Specificity: {100*specificity:.1f}%")
print(f"  PPV (Precision): {100*ppv:.1f}%")
print(f"  NPV: {100*npv:.1f}%")

# ============================================================================
# FEATURE IMPORTANCE
# ============================================================================
print("\n[7/8] Feature Importance Analysis...")

importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

print("\n  Top 10 Features:")
for i in range(min(10, len(existing_features))):
    print(f"    {i+1}. {existing_features[indices[i]]}: {importances[indices[i]]:.4f}")

# Save feature importance
fi_df = pd.DataFrame({
    'feature': existing_features,
    'importance': importances
}).sort_values('importance', ascending=False)
fi_df.to_csv('feature_importance.csv', index=False)

# ============================================================================
# GENERATE VISUALIZATIONS
# ============================================================================
print("\n[8/8] Generating Visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1. ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
axes[0, 0].plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {test_auc:.3f})')
axes[0, 0].plot([0, 1], [0, 1], 'k--', linewidth=1)
axes[0, 0].fill_between(fpr, tpr, alpha=0.3)
axes[0, 0].set_xlabel('False Positive Rate', fontsize=12)
axes[0, 0].set_ylabel('True Positive Rate', fontsize=12)
axes[0, 0].set_title('ROC Curve', fontsize=14, fontweight='bold')
axes[0, 0].legend(loc='lower right')
axes[0, 0].grid(True, alpha=0.3)

# 2. Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_prob)
axes[0, 1].plot(recall, precision, 'g-', linewidth=2, label=f'PR (AUC = {test_ap:.3f})')
axes[0, 1].fill_between(recall, precision, alpha=0.3, color='green')
axes[0, 1].set_xlabel('Recall', fontsize=12)
axes[0, 1].set_ylabel('Precision', fontsize=12)
axes[0, 1].set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
axes[0, 1].legend(loc='lower left')
axes[0, 1].grid(True, alpha=0.3)

# 3. Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0],
            xticklabels=['PASS', 'FAIL'], yticklabels=['PASS', 'FAIL'])
axes[1, 0].set_xlabel('Predicted', fontsize=12)
axes[1, 0].set_ylabel('Actual', fontsize=12)
axes[1, 0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')

# 4. Feature Importance
top_n = min(15, len(existing_features))
top_features = fi_df.head(top_n)
colors = plt.cm.viridis(np.linspace(0.2, 0.8, top_n))
axes[1, 1].barh(range(top_n), top_features['importance'].values[::-1], color=colors)
axes[1, 1].set_yticks(range(top_n))
axes[1, 1].set_yticklabels(top_features['feature'].values[::-1])
axes[1, 1].set_xlabel('Importance', fontsize=12)
axes[1, 1].set_title('Feature Importance (Top 15)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('ml_analysis_results.png', dpi=150, bbox_inches='tight')
plt.close()

print("  ✓ Saved: ml_analysis_results.png")

# Save ROC curve separately
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {test_auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
plt.fill_between(fpr, tpr, alpha=0.3)
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve - ABP Signal Integrity Classifier', fontsize=14, fontweight='bold')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.savefig('roc_curve.png', dpi=150, bbox_inches='tight')
plt.close()

print("  ✓ Saved: roc_curve.png")

# ============================================================================
# SAVE RESULTS SUMMARY
# ============================================================================

results = {
    'n_total': len(df),
    'n_train': len(X_train),
    'n_test': len(X_test),
    'n_pass': int((df['label'] == 'PASS').sum()),
    'n_fail': int((df['label'] == 'FAIL').sum()),
    'failure_rate': float(100 * (df['label'] == 'FAIL').mean()),
    'test_roc_auc': float(test_auc),
    'test_pr_auc': float(test_ap),
    'cv_mean_auc': float(cv_scores.mean()),
    'cv_std_auc': float(cv_scores.std()),
    'bootstrap_ci_lower': float(ci_lower),
    'bootstrap_ci_upper': float(ci_upper),
    'sensitivity': float(sensitivity),
    'specificity': float(specificity),
    'ppv': float(ppv),
    'npv': float(npv),
    'tn': int(tn),
    'fp': int(fp),
    'fn': int(fn),
    'tp': int(tp)
}

results_df = pd.DataFrame([results])
results_df.to_csv('ml_results_summary.csv', index=False)

print("\n" + "="*70)
print("ML ANALYSIS COMPLETE!")
print("="*70)

print(f"""
SUMMARY:
--------
Total samples: {results['n_total']}
FAIL rate: {results['failure_rate']:.1f}%

Model Performance:
  ROC-AUC: {results['test_roc_auc']:.3f}
  PR-AUC: {results['test_pr_auc']:.3f}
  95% Bootstrap CI: [{results['bootstrap_ci_lower']:.3f}, {results['bootstrap_ci_upper']:.3f}]
  
Cross-Validation:
  Mean AUC: {results['cv_mean_auc']:.3f} ± {results['cv_std_auc']:.3f}

Confusion Matrix:
  TN={results['tn']}, FP={results['fp']}
  FN={results['fn']}, TP={results['tp']}

Files Created:
  ✓ ml_results_summary.csv
  ✓ feature_importance.csv
  ✓ ml_analysis_results.png
  ✓ roc_curve.png
""")
