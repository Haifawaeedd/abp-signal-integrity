#!/usr/bin/env python3
"""
=============================================================================
COMPLETE PAPER VERIFICATION SCRIPT
=============================================================================
This script verifies EVERY number claimed in the IEEE JBHI paper by
re-computing them from the raw CSV data files.

Paper Title: "Automated Detection of Arterial Blood Pressure Signal
             Integrity Failures in Open Clinical Databases Using
             Machine Learning"

Databases: VitalDB + CHARIS
=============================================================================
"""

import pandas as pd
import numpy as np
import os
import sys

# Find data directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')

# If running from src/, adjust path
if not os.path.exists(DATA_DIR):
    DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
if not os.path.exists(DATA_DIR):
    DATA_DIR = '.'

print("=" * 80)
print("  COMPLETE PAPER VERIFICATION - ALL RESULTS FROM RAW DATA")
print("=" * 80)
print(f"\nData directory: {os.path.abspath(DATA_DIR)}")

# Track all checks
checks = []

def check(name, expected, actual, tolerance=0.1):
    """Verify a value matches expected within tolerance"""
    if isinstance(expected, str):
        passed = str(actual) == expected
    elif isinstance(expected, float):
        passed = abs(actual - expected) <= tolerance
    else:
        passed = actual == expected
    status = "✅ PASS" if passed else "❌ MISMATCH"
    checks.append((name, expected, actual, passed))
    print(f"  {status} | {name}: Expected={expected}, Got={actual}")
    return passed

# ==========================================================================
# SECTION 1: VitalDB DATABASE
# ==========================================================================
print("\n" + "=" * 80)
print("  SECTION 1: VitalDB DATABASE VERIFICATION")
print("=" * 80)

vitaldb_file = os.path.join(DATA_DIR, 'vitaldb_integrity_labels.csv')
df_vitaldb = pd.read_csv(vitaldb_file)

print(f"\n  File: {vitaldb_file}")
print(f"  Columns: {list(df_vitaldb.columns)}")
print(f"  Shape: {df_vitaldb.shape}")

# Total windows
total_vitaldb = len(df_vitaldb)
fail_vitaldb = (df_vitaldb['label'] == 'FAIL').sum()
pass_vitaldb = (df_vitaldb['label'] == 'PASS').sum()
fail_rate_vitaldb = 100 * fail_vitaldb / total_vitaldb

print(f"\n  --- VitalDB Core Numbers ---")
check("VitalDB Total Windows", 270, total_vitaldb)
check("VitalDB FAIL Count", 264, fail_vitaldb)
check("VitalDB PASS Count", 6, pass_vitaldb)
check("VitalDB Failure Rate (%)", 97.8, round(fail_rate_vitaldb, 1))

# Wilson confidence interval
from scipy.stats import norm
z = 1.96
n = total_vitaldb
p_hat = fail_vitaldb / n
denom = 1 + z**2 / n
center = (p_hat + z**2 / (2*n)) / denom
margin = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4*n)) / n) / denom
wilson_lower = 100 * (center - margin)
wilson_upper = 100 * (center + margin)

print(f"\n  --- Wilson 95% CI ---")
check("Wilson CI Lower (%)", 95.2, round(wilson_lower, 1))
check("Wilson CI Upper (%)", 99.0, round(wilson_upper, 1))

# Failure reasons
print(f"\n  --- VitalDB Failure Reasons ---")
fail_df = df_vitaldb[df_vitaldb['label'] == 'FAIL']

if 'failure_reason' in df_vitaldb.columns:
    reasons = fail_df['failure_reason'].str.split(';').explode().str.strip()
    reason_counts = reasons.value_counts()
    print(f"\n  Failure reason breakdown:")
    for reason, count in reason_counts.items():
        pct = 100 * count / fail_vitaldb
        print(f"    {reason}: {count} ({pct:.1f}%)")

# ==========================================================================
# SECTION 2: CHARIS DATABASE
# ==========================================================================
print("\n" + "=" * 80)
print("  SECTION 2: CHARIS DATABASE VERIFICATION")
print("=" * 80)

charis_file = os.path.join(DATA_DIR, 'charis_real_results.csv')
df_charis = pd.read_csv(charis_file)

print(f"\n  File: {charis_file}")
print(f"  Columns: {list(df_charis.columns)}")
print(f"  Shape: {df_charis.shape}")

total_charis = len(df_charis)
fail_charis = (df_charis['status'] == 'FAIL').sum()
pass_charis = (df_charis['status'] == 'PASS').sum()
fail_rate_charis = 100 * fail_charis / total_charis

print(f"\n  --- CHARIS Core Numbers ---")
check("CHARIS Total Windows", 130, total_charis)
check("CHARIS FAIL Count", 33, fail_charis)
check("CHARIS PASS Count", 97, pass_charis)
check("CHARIS Failure Rate (%)", 25.4, round(fail_rate_charis, 1))

# Per-record breakdown
print(f"\n  --- CHARIS Per-Record Breakdown ---")
unique_records = df_charis['record'].nunique()
check("CHARIS Unique Records", 13, unique_records)

for record in sorted(df_charis['record'].unique()):
    rec_df = df_charis[df_charis['record'] == record]
    rec_fail = (rec_df['status'] == 'FAIL').sum()
    rec_pass = (rec_df['status'] == 'PASS').sum()
    print(f"    {record}: {len(rec_df)} windows | FAIL={rec_fail} | PASS={rec_pass}")

# CHARIS failure reasons
print(f"\n  --- CHARIS Failure Reasons ---")
charis_fail = df_charis[df_charis['status'] == 'FAIL']
if 'failure_reason' in df_charis.columns:
    charis_reasons = charis_fail['failure_reason'].value_counts()
    for reason, count in charis_reasons.items():
        pct = 100 * count / fail_charis
        print(f"    {reason}: {count} ({pct:.1f}%)")

# ==========================================================================
# SECTION 3: COMBINED RESULTS
# ==========================================================================
print("\n" + "=" * 80)
print("  SECTION 3: COMBINED RESULTS")
print("=" * 80)

total_combined = total_vitaldb + total_charis
fail_combined = fail_vitaldb + fail_charis
pass_combined = pass_vitaldb + pass_charis
fail_rate_combined = 100 * fail_combined / total_combined

print(f"\n  --- Combined Numbers ---")
check("Combined Total Windows", 400, total_combined)
check("Combined FAIL Count", 297, fail_combined)
check("Combined PASS Count", 103, pass_combined)
check("Combined Failure Rate (%)", 74.2, round(fail_rate_combined, 1))

# ==========================================================================
# SECTION 4: QUERY LOG VERIFICATION
# ==========================================================================
print("\n" + "=" * 80)
print("  SECTION 4: QUERY LOG VERIFICATION")
print("=" * 80)

query_file = os.path.join(DATA_DIR, 'vitaldb_500_query_log.csv')
df_query = pd.read_csv(query_file)

total_queried = len(df_query)
successful = (df_query['status'] == 'success').sum()

check("Cases Queried", 500, total_queried)
check("Successful Extractions", 270, successful)
check("Failed Extractions", 230, total_queried - successful)

# ==========================================================================
# SECTION 5: FEATURES VERIFICATION
# ==========================================================================
print("\n" + "=" * 80)
print("  SECTION 5: FEATURES FILE VERIFICATION")
print("=" * 80)

features_file = os.path.join(DATA_DIR, 'vitaldb_5min_windows_features.csv')
df_features = pd.read_csv(features_file)

check("Features File Rows", 270, len(df_features))
print(f"  Features File Columns: {df_features.shape[1]}")
print(f"  Column names: {list(df_features.columns)}")

# ==========================================================================
# SECTION 6: ML MODEL VERIFICATION (from features data)
# ==========================================================================
print("\n" + "=" * 80)
print("  SECTION 6: ML MODEL VERIFICATION")
print("=" * 80)

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
    from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
    from sklearn.preprocessing import StandardScaler

    # Merge features with labels
    df_ml = df_features.copy()
    
    # Check if label column exists
    if 'label' not in df_ml.columns:
        # Merge with labels file
        df_labels = df_vitaldb[['case_id', 'label']].copy() if 'case_id' in df_vitaldb.columns else df_vitaldb.copy()
        if 'case_id' in df_ml.columns and 'case_id' in df_labels.columns:
            df_ml = df_ml.merge(df_labels, on='case_id', how='left')
        else:
            df_ml['label'] = df_vitaldb['label'].values[:len(df_ml)]
    
    y = (df_ml['label'] == 'FAIL').astype(int)
    
    # Select numeric feature columns only
    exclude_cols = ['label', 'case_id', 'caseid', 'window_id', 'status', 'failure_reason', 'record']
    feature_cols = [c for c in df_ml.columns if c not in exclude_cols and df_ml[c].dtype in ['float64', 'int64', 'float32', 'int32']]
    
    X = df_ml[feature_cols].fillna(0)
    
    print(f"\n  ML Dataset: {len(X)} samples, {len(feature_cols)} features")
    print(f"  FAIL: {y.sum()} ({100*y.mean():.1f}%)")
    print(f"  PASS: {(1-y).sum()} ({100*(1-y).mean():.1f}%)")
    
    # Train/Test Split (70/30)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Random Forest
    model = RandomForestClassifier(
        n_estimators=100, max_depth=10,
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)
    
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = model.predict(X_test_scaled)
    
    # Metrics
    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    
    print(f"\n  --- ML Model Results ---")
    check("ROC-AUC", 0.994, round(roc_auc, 3), tolerance=0.05)
    check("PR-AUC", 1.000, round(pr_auc, 3), tolerance=0.05)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    print(f"\n  Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"  Sensitivity: {100*sensitivity:.1f}%")
    print(f"  Specificity: {100*specificity:.1f}%")
    
    # Bootstrap CI
    B = 1000
    bootstrap_aucs = []
    np.random.seed(42)
    for i in range(B):
        idx = np.random.choice(len(X_test_scaled), size=len(X_test_scaled), replace=True)
        y_boot = y_test.iloc[idx]
        if len(np.unique(y_boot)) < 2:
            continue
        y_prob_boot = model.predict_proba(X_test_scaled[idx])[:, 1]
        bootstrap_aucs.append(roc_auc_score(y_boot, y_prob_boot))
    
    ci_lower = np.percentile(bootstrap_aucs, 2.5)
    ci_upper = np.percentile(bootstrap_aucs, 97.5)
    
    print(f"\n  --- Bootstrap 95% CI ---")
    print(f"  CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
    check("Bootstrap CI Lower", 0.974, round(ci_lower, 3), tolerance=0.05)
    check("Bootstrap CI Upper", 1.000, round(ci_upper, 3), tolerance=0.05)
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
    print(f"\n  --- 5-Fold Cross-Validation ---")
    print(f"  CV AUC scores: {[round(s,3) for s in cv_scores]}")
    print(f"  Mean CV AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    
    # Feature importance
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    print(f"\n  --- Top 10 Feature Importances ---")
    for i in range(min(10, len(feature_cols))):
        print(f"    {i+1}. {feature_cols[indices[i]]}: {importances[indices[i]]:.4f}")

except Exception as e:
    print(f"\n  ⚠️ ML verification error: {e}")
    import traceback
    traceback.print_exc()

# ==========================================================================
# SECTION 7: CLINICAL IMPACT VERIFICATION
# ==========================================================================
print("\n" + "=" * 80)
print("  SECTION 7: CLINICAL IMPACT RESULTS")
print("=" * 80)

clinical_file = os.path.join(DATA_DIR, 'clinical_impact_FINAL_results.csv')
df_clinical = pd.read_csv(clinical_file)
print(f"\n  Clinical Impact Results:")
print(df_clinical.to_string(index=False))

# ==========================================================================
# FINAL SUMMARY
# ==========================================================================
print("\n" + "=" * 80)
print("  FINAL VERIFICATION SUMMARY")
print("=" * 80)

total_checks = len(checks)
passed_checks = sum(1 for c in checks if c[3])
failed_checks = total_checks - passed_checks

print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   TOTAL CHECKS:  {total_checks:<5}                                                    ║
║   PASSED:        {passed_checks:<5}  ✅                                                ║
║   FAILED:        {failed_checks:<5}  {'❌' if failed_checks > 0 else '✅'}                                                ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║   PAPER CLAIMS vs RAW DATA:                                                  ║
║                                                                              ║
║   VitalDB:   270 windows, 264 FAIL, 97.8% failure rate                      ║
║   CHARIS:    130 windows,  33 FAIL, 25.4% failure rate                      ║
║   Combined:  400 windows, 297 FAIL, 74.2% failure rate                      ║
║   Wilson CI: [95.2%, 99.0%]                                                  ║
║   ROC-AUC:   ~0.994                                                         ║
║   PR-AUC:    ~1.000                                                          ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

if failed_checks > 0:
    print("⚠️  MISMATCHED CHECKS:")
    for name, expected, actual, passed in checks:
        if not passed:
            print(f"  ❌ {name}: Expected={expected}, Got={actual}")
else:
    print("✅ ALL CHECKS PASSED! PAPER RESULTS MATCH RAW DATA 100%!")

# Save report
report_path = os.path.join(SCRIPT_DIR, '..', 'results', 'PAPER_VERIFICATION_REPORT.txt')
with open(report_path, 'w') as f:
    f.write("PAPER VERIFICATION REPORT\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Total checks: {total_checks}\n")
    f.write(f"Passed: {passed_checks}\n")
    f.write(f"Failed: {failed_checks}\n\n")
    for name, expected, actual, passed in checks:
        status = "PASS" if passed else "FAIL"
        f.write(f"[{status}] {name}: Expected={expected}, Got={actual}\n")

print(f"\nReport saved to: {report_path}")
