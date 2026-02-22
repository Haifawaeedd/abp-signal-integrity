"""
COMPREHENSIVE COMPARISON: Paper vs GitHub Code vs CSV Data
==========================================================
This script runs all analyses and compares results with paper claims.
"""

import pandas as pd
import numpy as np
from scipy.stats import binom
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, recall_score
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("  COMPREHENSIVE VERIFICATION: Paper vs Code vs Data")
print("="*70)

# ============================================================
# 1. VITALDB DATA VERIFICATION
# ============================================================
print("\n" + "="*70)
print("  1. VitalDB DATA VERIFICATION")
print("="*70)

df_labels = pd.read_csv('/home/ubuntu/abp-signal-integrity/data/vitaldb_integrity_labels.csv')
df_query = pd.read_csv('/home/ubuntu/abp-signal-integrity/data/vitaldb_500_query_log.csv')
df_features = pd.read_csv('/home/ubuntu/abp-signal-integrity/data/vitaldb_5min_windows_features.csv')

total_windows = len(df_labels)
fail_count = (df_labels['label'] == 'FAIL').sum()
pass_count = (df_labels['label'] == 'PASS').sum()
fail_rate = 100 * fail_count / total_windows
pass_rate = 100 * pass_count / total_windows

cases_queried = len(df_query)
cases_successful = (df_query['status'] == 'success').sum()

# Wilson CI
n = total_windows
p_hat = fail_count / n
z = 1.96
denom = 1 + z**2/n
center = (p_hat + z**2/(2*n)) / denom
margin = z * np.sqrt((p_hat*(1-p_hat) + z**2/(4*n)) / n) / denom
wilson_lower = 100 * (center - margin)
wilson_upper = 100 * (center + margin)

print(f"\nCases queried:        {cases_queried}")
print(f"Cases successful:     {cases_successful}")
print(f"Total windows:        {total_windows}")
print(f"FAIL:                 {fail_count} ({fail_rate:.1f}%)")
print(f"PASS:                 {pass_count} ({pass_rate:.1f}%)")
print(f"Wilson 95% CI:        [{wilson_lower:.1f}%, {wilson_upper:.1f}%]")

# Failure reasons
print(f"\nFailure Reasons (from labels CSV):")
fail_df = df_labels[df_labels['label'] == 'FAIL']
if 'failure_reason' in fail_df.columns:
    reasons = fail_df['failure_reason'].str.split(';').explode().str.strip()
    for reason, count in reasons.value_counts().items():
        print(f"  {reason}: {count}/{fail_count} ({100*count/fail_count:.1f}%)")

# Flatline count
if 'failure_reason' in df_labels.columns:
    flatline_count = df_labels['failure_reason'].str.contains('flatline|Flatline|FLATLINE', na=False).sum()
    print(f"\nFlatline detected in: {flatline_count} of {total_windows} ({100*flatline_count/total_windows:.1f}%)")

# ============================================================
# 2. CHARIS DATA VERIFICATION
# ============================================================
print("\n" + "="*70)
print("  2. CHARIS DATA VERIFICATION")
print("="*70)

df_charis = pd.read_csv('/home/ubuntu/abp-signal-integrity/data/charis_real_results.csv')

charis_total = len(df_charis)
charis_fail = (df_charis['status'] == 'FAIL').sum()
charis_pass = (df_charis['status'] == 'PASS').sum()
charis_fail_rate = 100 * charis_fail / charis_total
charis_pass_rate = 100 * charis_pass / charis_total

# Unique records
if 'record_id' in df_charis.columns:
    charis_records = df_charis['record_id'].nunique()
elif 'record' in df_charis.columns:
    charis_records = df_charis['record'].nunique()
else:
    charis_records = "Unknown"

print(f"\nTotal windows:        {charis_total}")
print(f"Unique records:       {charis_records}")
print(f"FAIL:                 {charis_fail} ({charis_fail_rate:.1f}%)")
print(f"PASS:                 {charis_pass} ({charis_pass_rate:.1f}%)")

# CHARIS failure reasons
charis_fail_df = df_charis[df_charis['status'] == 'FAIL']
if 'failure_reason' in charis_fail_df.columns:
    print(f"\nCHARIS Failure Reasons:")
    ch_reasons = charis_fail_df['failure_reason'].str.split(';').explode().str.strip()
    for reason, count in ch_reasons.value_counts().items():
        print(f"  {reason}: {count}/{charis_fail} ({100*count/charis_fail:.1f}%)")

# ============================================================
# 3. ML MODEL VERIFICATION
# ============================================================
print("\n" + "="*70)
print("  3. ML MODEL VERIFICATION")
print("="*70)

# Prepare features
feature_cols = [c for c in df_features.columns if c not in ['case_id', 'caseid', 'label', 'window_id', 'failure_reason', 'status']]
X = df_features[feature_cols].copy()
X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

# Get labels
if 'label' in df_features.columns:
    y = (df_features['label'] == 'FAIL').astype(int)
elif 'label' in df_labels.columns:
    y = (df_labels['label'] == 'FAIL').astype(int)

print(f"\nFeatures shape: {X.shape}")
print(f"Number of features: {X.shape[1]}")
print(f"Feature names: {list(X.columns)}")

# Train/test split (stratified, same as paper: ~70/30)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\nTrain size: {len(X_train)}")
print(f"Test size:  {len(X_test)}")
print(f"Test FAIL:  {y_test.sum()}")
print(f"Test PASS:  {(y_test == 0).sum()}")

# Train Random Forest
rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
rf.fit(X_train, y_train)

y_prob = rf.predict_proba(X_test)[:, 1]
y_pred = rf.predict(X_test)

roc_auc = roc_auc_score(y_test, y_prob)
pr_auc = average_precision_score(y_test, y_prob)

print(f"\nROC-AUC:    {roc_auc:.3f}")
print(f"PR-AUC:     {pr_auc:.3f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(f"\nConfusion Matrix:")
print(f"  TN={cm[0][0]}, FP={cm[0][1]}")
print(f"  FN={cm[1][0]}, TP={cm[1][1]}")

sensitivity = recall_score(y_test, y_pred)
if cm[0][0] + cm[0][1] > 0:
    specificity = cm[0][0] / (cm[0][0] + cm[0][1])
else:
    specificity = 0
print(f"Sensitivity: {sensitivity:.3f}")
print(f"Specificity: {specificity:.3f}")

# Bootstrap CI
print(f"\nBootstrap CI (1000 iterations):")
n_boot = 1000
boot_roc = []
for i in range(n_boot):
    idx = np.random.choice(len(y_test), len(y_test), replace=True)
    if len(np.unique(y_test.values[idx])) < 2:
        continue
    boot_roc.append(roc_auc_score(y_test.values[idx], y_prob[idx]))

boot_roc = np.array(boot_roc)
print(f"  Mean ROC-AUC: {boot_roc.mean():.3f}")
print(f"  95% CI: [{np.percentile(boot_roc, 2.5):.3f}, {np.percentile(boot_roc, 97.5):.3f}]")

# Feature importance
importances = rf.feature_importances_
feat_imp = sorted(zip(X.columns, importances), key=lambda x: -x[1])
print(f"\nTop 10 Feature Importances:")
for name, imp in feat_imp[:10]:
    print(f"  {name}: {imp:.4f}")

# Nested CV
print(f"\nNested Cross-Validation (5-fold):")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf, X, y, cv=cv, scoring='roc_auc')
print(f"  Mean ROC-AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# 50 repeated splits
print(f"\n50 Repeated Stratified Splits:")
split_aucs = []
for i in range(50):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=i, stratify=y)
    rf_temp = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
    rf_temp.fit(Xtr, ytr)
    yp = rf_temp.predict_proba(Xte)[:, 1]
    if len(np.unique(yte)) >= 2:
        split_aucs.append(roc_auc_score(yte, yp))
split_aucs = np.array(split_aucs)
print(f"  Mean: {split_aucs.mean():.3f}")
print(f"  Std:  {split_aucs.std():.3f}")
print(f"  Min:  {split_aucs.min():.3f}")
print(f"  Max:  {split_aucs.max():.3f}")

# ============================================================
# 4. COMBINED RESULTS
# ============================================================
print("\n" + "="*70)
print("  4. COMBINED RESULTS")
print("="*70)

combined_total = total_windows + charis_total
combined_fail = fail_count + charis_fail
combined_pass = pass_count + charis_pass
print(f"\nCombined Total:  {combined_total}")
print(f"Combined FAIL:   {combined_fail} ({100*combined_fail/combined_total:.1f}%)")
print(f"Combined PASS:   {combined_pass} ({100*combined_pass/combined_total:.1f}%)")

# ============================================================
# 5. COMPARISON WITH PAPER
# ============================================================
print("\n" + "="*70)
print("  5. COMPARISON: PAPER vs ACTUAL DATA")
print("="*70)

comparisons = [
    ("VitalDB Total Windows", "270", str(total_windows)),
    ("VitalDB FAIL", "264 (97.8%)", f"{fail_count} ({fail_rate:.1f}%)"),
    ("VitalDB PASS", "6 (2.2%)", f"{pass_count} ({pass_rate:.1f}%)"),
    ("Wilson CI", "[95.3%, 99.1%]", f"[{wilson_lower:.1f}%, {wilson_upper:.1f}%]"),
    ("Cases Queried", "500", str(cases_queried)),
    ("Cases Successful", "270", str(cases_successful)),
    ("CHARIS Total", "130", str(charis_total)),
    ("CHARIS FAIL", "33 (25.4%)", f"{charis_fail} ({charis_fail_rate:.1f}%)"),
    ("CHARIS PASS", "97 (74.6%)", f"{charis_pass} ({charis_pass_rate:.1f}%)"),
    ("CHARIS Records", "13", str(charis_records)),
    ("ROC-AUC (paper text)", "0.994", f"{roc_auc:.3f}"),
    ("PR-AUC", "1.000", f"{pr_auc:.3f}"),
    ("Nested CV ROC-AUC", "0.996", f"{cv_scores.mean():.3f}"),
    ("50-split mean", "0.995", f"{split_aucs.mean():.3f}"),
    ("Combined Total", "400", str(combined_total)),
    ("Combined FAIL", "297 (74.2%)", f"{combined_fail} ({100*combined_fail/combined_total:.1f}%)"),
]

print(f"\n{'Metric':<30} {'Paper':>20} {'Actual':>20} {'Match':>8}")
print("-"*80)
for metric, paper_val, actual_val in comparisons:
    match = "✅" if paper_val.strip() == actual_val.strip() else "⚠️"
    print(f"{metric:<30} {paper_val:>20} {actual_val:>20} {match:>8}")

# ============================================================
# 6. ERRORS FOUND IN PAPER
# ============================================================
print("\n" + "="*70)
print("  6. ERRORS FOUND IN PAPER (MUST FIX)")
print("="*70)

print("""
ERROR 1: Methods Section III.A
  Paper says: "279 cases were retained"
  Should be:  "270 cases were retained" (matches Table I and Results)

ERROR 2: Discussion Section V.B  
  Paper says: "25.4% vs. 95.0%"
  Should be:  "25.4% vs. 97.8%"

ERROR 3: Fig 2 caption inconsistency
  Fig 2 caption says: ROC-AUC = 0.970, bootstrap = 0.971 [0.914, 1.000]
  Text says:          ROC-AUC = 0.994, bootstrap = 0.994 [0.974, 1.000]
  → Must be consistent. Use the value from the actual model run.

ERROR 4: Fig 3 caption inconsistency
  Fig 3 caption says: PR-AUC = 0.998
  Text says:          PR-AUC = 1.000
  → Must be consistent.

ERROR 5: Fig 4 caption inconsistency
  Fig 4 caption says: recall = 0.875
  Text says:          recall = 0.889
  → Must be consistent.
""")

print("="*70)
print("  VERIFICATION COMPLETE")
print("="*70)
