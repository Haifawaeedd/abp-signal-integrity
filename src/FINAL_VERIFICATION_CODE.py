#!/usr/bin/env python3
"""
=============================================================================
FINAL VERIFICATION CODE - COMPLETE PROOF OF RESULTS
=============================================================================
This script verifies ALL results from BOTH databases.
Run this code yourself to confirm the numbers are accurate.

Author: Verification Script
Date: February 2026
=============================================================================
"""

import pandas as pd
import os

print("=" * 80)
print("🔬 FINAL VERIFICATION CODE - COMPLETE PROOF")
print("=" * 80)
print(f"\nRunning from: {os.getcwd()}")
print(f"Date: February 3, 2026")

# =============================================================================
# STEP 1: CHECK FILES EXIST
# =============================================================================
print("\n" + "=" * 80)
print("STEP 1: CHECKING FILES EXIST")
print("=" * 80)

files = {
    'VitalDB Labels': 'vitaldb_integrity_labels.csv',
    'CHARIS Results': 'charis_real_results.csv',
    'Clinical Impact': 'clinical_impact_FINAL_results.csv'
}

for name, filename in files.items():
    exists = os.path.exists(filename)
    size = os.path.getsize(filename) if exists else 0
    status = "✅ EXISTS" if exists else "❌ MISSING"
    print(f"  {name}: {status} ({size:,} bytes)")

# =============================================================================
# STEP 2: VITALDB VERIFICATION
# =============================================================================
print("\n" + "=" * 80)
print("STEP 2: VITALDB VERIFICATION")
print("=" * 80)

print("\n📂 Reading: vitaldb_integrity_labels.csv")
df_vitaldb = pd.read_csv('vitaldb_integrity_labels.csv')

print(f"\n📋 File Structure:")
print(f"   Columns: {list(df_vitaldb.columns)}")
print(f"   Total Rows: {len(df_vitaldb)}")

print(f"\n📊 First 5 rows:")
print(df_vitaldb.head().to_string())

print(f"\n📊 Last 5 rows:")
print(df_vitaldb.tail().to_string())

print(f"\n📊 Label Value Counts (raw pandas output):")
label_counts = df_vitaldb['label'].value_counts()
print(label_counts)

# Manual calculation
v_total = len(df_vitaldb)
v_fail = (df_vitaldb['label'] == 'FAIL').sum()
v_pass = (df_vitaldb['label'] == 'PASS').sum()
v_other = v_total - v_fail - v_pass

print(f"\n✅ VITALDB MANUAL COUNT:")
print(f"   Total rows in file: {v_total}")
print(f"   Rows with label='FAIL': {v_fail}")
print(f"   Rows with label='PASS': {v_pass}")
print(f"   Other labels: {v_other}")
print(f"   FAILURE RATE: {v_fail}/{v_total} = {100*v_fail/v_total:.2f}%")

# =============================================================================
# STEP 3: CHARIS VERIFICATION
# =============================================================================
print("\n" + "=" * 80)
print("STEP 3: CHARIS VERIFICATION")
print("=" * 80)

print("\n📂 Reading: charis_real_results.csv")
df_charis = pd.read_csv('charis_real_results.csv')

print(f"\n📋 File Structure:")
print(f"   Columns: {list(df_charis.columns)}")
print(f"   Total Rows: {len(df_charis)}")

print(f"\n📊 First 5 rows (selected columns):")
print(df_charis[['record', 'window', 'status', 'failure_reason']].head().to_string())

print(f"\n📊 Last 5 rows (selected columns):")
print(df_charis[['record', 'window', 'status', 'failure_reason']].tail().to_string())

print(f"\n📊 Status Value Counts (raw pandas output):")
status_counts = df_charis['status'].value_counts()
print(status_counts)

# Manual calculation
c_total = len(df_charis)
c_fail = (df_charis['status'] == 'FAIL').sum()
c_pass = (df_charis['status'] == 'PASS').sum()
c_other = c_total - c_fail - c_pass

print(f"\n✅ CHARIS MANUAL COUNT:")
print(f"   Total rows in file: {c_total}")
print(f"   Rows with status='FAIL': {c_fail}")
print(f"   Rows with status='PASS': {c_pass}")
print(f"   Other status: {c_other}")
print(f"   FAILURE RATE: {c_fail}/{c_total} = {100*c_fail/c_total:.2f}%")

# Per-record breakdown
print(f"\n📊 CHARIS Per-Record Breakdown:")
print("-" * 50)
for record in sorted(df_charis['record'].unique()):
    rec_df = df_charis[df_charis['record'] == record]
    rec_fail = (rec_df['status'] == 'FAIL').sum()
    rec_pass = (rec_df['status'] == 'PASS').sum()
    print(f"   {record}: {len(rec_df)} windows | FAIL={rec_fail} | PASS={rec_pass}")

# =============================================================================
# STEP 4: COMBINED RESULTS
# =============================================================================
print("\n" + "=" * 80)
print("STEP 4: COMBINED RESULTS")
print("=" * 80)

combined_total = v_total + c_total
combined_fail = v_fail + c_fail
combined_pass = v_pass + c_pass

print(f"\n📊 COMBINED CALCULATION:")
print(f"   VitalDB: {v_total} windows ({v_fail} FAIL + {v_pass} PASS)")
print(f"   CHARIS:  {c_total} windows ({c_fail} FAIL + {c_pass} PASS)")
print(f"   -" * 30)
print(f"   TOTAL:   {combined_total} windows ({combined_fail} FAIL + {combined_pass} PASS)")
print(f"   COMBINED FAILURE RATE: {combined_fail}/{combined_total} = {100*combined_fail/combined_total:.2f}%")

# =============================================================================
# STEP 5: CLINICAL IMPACT VERIFICATION
# =============================================================================
print("\n" + "=" * 80)
print("STEP 5: CLINICAL IMPACT VERIFICATION")
print("=" * 80)

print("\n📂 Reading: clinical_impact_FINAL_results.csv")
df_clinical = pd.read_csv('clinical_impact_FINAL_results.csv')

print(f"\n📊 Clinical Impact Results:")
print(df_clinical.to_string())

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("=" * 80)
print("                    FINAL VERIFIED RESULTS")
print("=" * 80)
print("=" * 80)

print(f"""

╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   DATABASE 1: VitalDB (Korea - Surgical)                                     ║
║   ─────────────────────────────────────                                      ║
║   Source File: vitaldb_integrity_labels.csv                                  ║
║   Total Windows: {v_total}                                                       ║
║   FAIL: {v_fail} ({100*v_fail/v_total:.1f}%)                                                            ║
║   PASS: {v_pass} ({100*v_pass/v_total:.1f}%)                                                             ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║   DATABASE 2: CHARIS                                                         ║
║   ─────────────────────────────────────                                      ║
║   Source File: charis_real_results.csv                                       ║
║   Total Windows: {c_total}                                                       ║
║   FAIL: {c_fail} ({100*c_fail/c_total:.1f}%)                                                            ║
║   PASS: {c_pass} ({100*c_pass/c_total:.1f}%)                                                           ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║   COMBINED RESULTS                                                           ║
║   ─────────────────────────────────────                                      ║
║   Total Windows: {combined_total}                                                       ║
║   FAIL: {combined_fail} ({100*combined_fail/combined_total:.1f}%)                                                           ║
║   PASS: {combined_pass} ({100*combined_pass/combined_total:.1f}%)                                                           ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

✅ VERIFICATION COMPLETE!
✅ ALL NUMBERS ARE ACCURATE AND DERIVED FROM RAW CSV FILES!
✅ YOU CAN RUN THIS CODE YOURSELF TO CONFIRM!

""")

# Save results to file
with open('VERIFICATION_RESULTS.txt', 'w') as f:
    f.write("FINAL VERIFICATION RESULTS\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"VitalDB:\n")
    f.write(f"  Total: {v_total}\n")
    f.write(f"  FAIL: {v_fail} ({100*v_fail/v_total:.1f}%)\n")
    f.write(f"  PASS: {v_pass} ({100*v_pass/v_total:.1f}%)\n\n")
    f.write(f"CHARIS:\n")
    f.write(f"  Total: {c_total}\n")
    f.write(f"  FAIL: {c_fail} ({100*c_fail/c_total:.1f}%)\n")
    f.write(f"  PASS: {c_pass} ({100*c_pass/c_total:.1f}%)\n\n")
    f.write(f"COMBINED:\n")
    f.write(f"  Total: {combined_total}\n")
    f.write(f"  FAIL: {combined_fail} ({100*combined_fail/combined_total:.1f}%)\n")
    f.write(f"  PASS: {combined_pass} ({100*combined_pass/combined_total:.1f}%)\n")

print("📄 Results saved to: VERIFICATION_RESULTS.txt")
print("=" * 80)
