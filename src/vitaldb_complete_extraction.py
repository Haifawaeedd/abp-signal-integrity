#!/usr/bin/env python3
"""
VitalDB ABP Signal Integrity Analysis - Complete Extraction
============================================================
Purpose: Extract 500 five-minute ABP windows from VitalDB with full feature set
Output: features_vitaldb.csv (ready for ML analysis)

Author: Haifa's Research Team
Date: January 2025
"""

import vitaldb
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm
import warnings
import time
warnings.filterwarnings('ignore')

print("="*70)
print("VitalDB ABP SIGNAL INTEGRITY ANALYSIS")
print("Complete Data Extraction from Real VitalDB API")
print("="*70)

# ============================================================================
# CONFIGURATION
# ============================================================================

N_CASES_TARGET = 500  # Target number of cases to request
START_CASE = 1
WAVEFORM_FREQ = 500   # Hz (ABP waveform)
NUMERIC_FREQ = 2      # Hz (vital signs)
WINDOW_DURATION = 300 # seconds (5 minutes)

# Track names in VitalDB
TRACK_ABP_WAVE = 'SNUADC/ART'
TRACKS_NUMERICS = [
    'Solar8000/ART_SBP',
    'Solar8000/ART_DBP', 
    'Solar8000/ART_MBP'
]

# Physiologic thresholds (from clinical literature)
THRESHOLDS = {
    'SBP_range': (80, 200),
    'DBP_range': (40, 140),
    'MAP_range': (60, 120),
    'PP_range': (20, 100),
    'PP_extreme_low': 10,
    'PP_extreme_high': 150,
    'spike_threshold': 40
}

print(f"\nConfiguration:")
print(f"  Target cases: {N_CASES_TARGET}")
print(f"  Window duration: {WINDOW_DURATION} seconds (5 minutes)")
print(f"  Waveform sampling: {WAVEFORM_FREQ} Hz")
print(f"  Numeric sampling: {NUMERIC_FREQ} Hz")

# ============================================================================
# FEATURE EXTRACTION FUNCTIONS
# ============================================================================

def extract_time_domain(abp_wave):
    """Extract 6 time-domain features"""
    features = {}
    valid = abp_wave[~np.isnan(abp_wave)]
    
    if len(valid) == 0:
        return {k: np.nan for k in ['ABP_mean', 'ABP_std', 'ABP_min', 
                                     'ABP_max', 'ABP_range', 'ABP_IQR']}
    
    features['ABP_mean'] = float(np.mean(valid))
    features['ABP_std'] = float(np.std(valid))
    features['ABP_min'] = float(np.min(valid))
    features['ABP_max'] = float(np.max(valid))
    features['ABP_range'] = features['ABP_max'] - features['ABP_min']
    features['ABP_IQR'] = float(np.percentile(valid, 75) - np.percentile(valid, 25))
    
    return features


def extract_integrity(abp_wave):
    """Extract 6 signal integrity features"""
    features = {}
    total_samples = len(abp_wave)
    valid = abp_wave[~np.isnan(abp_wave)]
    
    # Negative values (physiologically impossible)
    features['negative_value_ratio'] = float(np.mean(abp_wave < 0)) if total_samples > 0 else np.nan
    
    # Zero values
    features['zero_value_ratio'] = float(np.mean(abp_wave == 0)) if total_samples > 0 else np.nan
    
    # Flatline detection (consecutive identical values)
    if len(valid) > 1:
        diff = np.diff(valid)
        flatline_samples = np.sum(diff == 0)
        features['flatline_ratio'] = float(flatline_samples / len(valid))
    else:
        features['flatline_ratio'] = np.nan
    
    # Spike detection (sudden changes > 50 mmHg)
    if len(valid) > 1:
        diff = np.abs(np.diff(valid))
        features['spike_ratio'] = float(np.mean(diff > 50))
    else:
        features['spike_ratio'] = np.nan
    
    # Saturation (values at sensor limits)
    features['saturation_ratio'] = float(np.mean((abp_wave <= 0) | (abp_wave >= 300))) if total_samples > 0 else np.nan
    
    # Missing data
    features['missing_ratio'] = float(np.mean(np.isnan(abp_wave))) if total_samples > 0 else np.nan
    
    return features


def extract_physiologic(sbp, dbp, map_values):
    """Extract 8 physiologic plausibility features"""
    features = {}
    
    if sbp is None or dbp is None or len(sbp) == 0 or len(dbp) == 0:
        return {k: np.nan for k in ['PP_mean', 'PP_std', 'PP_violation_ratio',
                                     'MAP_violation_ratio', 'SBP_violation_ratio',
                                     'DBP_violation_ratio', 'PP_negative_ratio',
                                     'PP_extreme_ratio']}
    
    # Remove NaN
    sbp_clean = sbp[~np.isnan(sbp)]
    dbp_clean = dbp[~np.isnan(dbp)]
    
    if len(sbp_clean) == 0 or len(dbp_clean) == 0:
        return {k: np.nan for k in ['PP_mean', 'PP_std', 'PP_violation_ratio',
                                     'MAP_violation_ratio', 'SBP_violation_ratio',
                                     'DBP_violation_ratio', 'PP_negative_ratio',
                                     'PP_extreme_ratio']}
    
    # Calculate pulse pressure
    min_len = min(len(sbp_clean), len(dbp_clean))
    pp = sbp_clean[:min_len] - dbp_clean[:min_len]
    
    features['PP_mean'] = float(np.mean(pp))
    features['PP_std'] = float(np.std(pp))
    
    # Pulse pressure violations
    pp_low, pp_high = THRESHOLDS['PP_range']
    features['PP_violation_ratio'] = float(np.mean((pp < pp_low) | (pp > pp_high)))
    
    # MAP violations
    if map_values is not None:
        map_clean = map_values[~np.isnan(map_values)]
        if len(map_clean) > 0:
            map_low, map_high = THRESHOLDS['MAP_range']
            features['MAP_violation_ratio'] = float(np.mean((map_clean < map_low) | (map_clean > map_high)))
        else:
            features['MAP_violation_ratio'] = np.nan
    else:
        features['MAP_violation_ratio'] = np.nan
    
    # SBP violations
    sbp_low, sbp_high = THRESHOLDS['SBP_range']
    features['SBP_violation_ratio'] = float(np.mean((sbp_clean < sbp_low) | (sbp_clean > sbp_high)))
    
    # DBP violations
    dbp_low, dbp_high = THRESHOLDS['DBP_range']
    features['DBP_violation_ratio'] = float(np.mean((dbp_clean < dbp_low) | (dbp_clean > dbp_high)))
    
    # Negative pulse pressure (SBP < DBP - impossible)
    features['PP_negative_ratio'] = float(np.mean(pp < 0))
    
    # Extreme pulse pressure
    features['PP_extreme_ratio'] = float(np.mean((pp < THRESHOLDS['PP_extreme_low']) | 
                                                  (pp > THRESHOLDS['PP_extreme_high'])))
    
    return features


def extract_alignment(abp_wave, sbp, dbp, map_values):
    """Extract 5 waveform-numeric alignment features"""
    features = {}
    
    # Downsample waveform to match numeric frequency
    samples_per_numeric = int(WAVEFORM_FREQ / NUMERIC_FREQ)
    n_windows = len(abp_wave) // samples_per_numeric
    
    if n_windows == 0:
        return {k: np.nan for k in ['r_MAP', 'MAD_MAP', 'r_SBP', 'r_DBP', 'alignment_score']}
    
    # Calculate waveform-derived values
    wave_means = []
    wave_maxs = []
    wave_mins = []
    
    for i in range(n_windows):
        start = i * samples_per_numeric
        end = start + samples_per_numeric
        segment = abp_wave[start:end]
        valid = segment[~np.isnan(segment)]
        
        if len(valid) > 0:
            wave_means.append(np.mean(valid))
            wave_maxs.append(np.max(valid))
            wave_mins.append(np.min(valid))
        else:
            wave_means.append(np.nan)
            wave_maxs.append(np.nan)
            wave_mins.append(np.nan)
    
    wave_means = np.array(wave_means)
    wave_maxs = np.array(wave_maxs)
    wave_mins = np.array(wave_mins)
    
    # Correlation with MAP
    if map_values is not None and len(map_values) > 0:
        min_len = min(len(wave_means), len(map_values))
        wm = wave_means[:min_len]
        mv = map_values[:min_len]
        
        # Remove NaN pairs
        mask = ~(np.isnan(wm) | np.isnan(mv))
        if mask.sum() > 10:
            r, _ = stats.pearsonr(wm[mask], mv[mask])
            features['r_MAP'] = float(r) if not np.isnan(r) else np.nan
            features['MAD_MAP'] = float(np.mean(np.abs(wm[mask] - mv[mask])))
        else:
            features['r_MAP'] = np.nan
            features['MAD_MAP'] = np.nan
    else:
        features['r_MAP'] = np.nan
        features['MAD_MAP'] = np.nan
    
    # Correlation with SBP
    if sbp is not None and len(sbp) > 0:
        min_len = min(len(wave_maxs), len(sbp))
        wmax = wave_maxs[:min_len]
        sv = sbp[:min_len]
        
        mask = ~(np.isnan(wmax) | np.isnan(sv))
        if mask.sum() > 10:
            r, _ = stats.pearsonr(wmax[mask], sv[mask])
            features['r_SBP'] = float(r) if not np.isnan(r) else np.nan
        else:
            features['r_SBP'] = np.nan
    else:
        features['r_SBP'] = np.nan
    
    # Correlation with DBP
    if dbp is not None and len(dbp) > 0:
        min_len = min(len(wave_mins), len(dbp))
        wmin = wave_mins[:min_len]
        dv = dbp[:min_len]
        
        mask = ~(np.isnan(wmin) | np.isnan(dv))
        if mask.sum() > 10:
            r, _ = stats.pearsonr(wmin[mask], dv[mask])
            features['r_DBP'] = float(r) if not np.isnan(r) else np.nan
        else:
            features['r_DBP'] = np.nan
    else:
        features['r_DBP'] = np.nan
    
    # Overall alignment score
    correlations = [features.get('r_MAP', np.nan), features.get('r_SBP', np.nan), features.get('r_DBP', np.nan)]
    valid_corrs = [c for c in correlations if not np.isnan(c)]
    features['alignment_score'] = float(np.mean(valid_corrs)) if valid_corrs else np.nan
    
    return features


def generate_label(features):
    """Generate PASS/FAIL label based on predefined integrity criteria"""
    
    fail_conditions = [
        # Negative values > 10%
        features.get('negative_value_ratio', 0) > 0.10,
        # Flatline > 30%
        features.get('flatline_ratio', 0) > 0.30,
        # Pulse pressure violations > 50%
        features.get('PP_violation_ratio', 0) > 0.50 if not np.isnan(features.get('PP_violation_ratio', np.nan)) else False,
        # MAP violations > 50%
        features.get('MAP_violation_ratio', 0) > 0.50 if not np.isnan(features.get('MAP_violation_ratio', np.nan)) else False,
        # Spikes > 10%
        features.get('spike_ratio', 0) > 0.10,
        # Saturation > 20%
        features.get('saturation_ratio', 0) > 0.20,
        # Missing > 30%
        features.get('missing_ratio', 0) > 0.30,
    ]
    
    return 'FAIL' if any(fail_conditions) else 'PASS'


def get_failure_reasons(features):
    """Get list of failure reasons for a case"""
    reasons = []
    
    if features.get('negative_value_ratio', 0) > 0.10:
        reasons.append('negative_values')
    if features.get('flatline_ratio', 0) > 0.30:
        reasons.append('extended_flatline')
    if features.get('PP_violation_ratio', 0) > 0.50 if not np.isnan(features.get('PP_violation_ratio', np.nan)) else False:
        reasons.append('PP_violation')
    if features.get('MAP_violation_ratio', 0) > 0.50 if not np.isnan(features.get('MAP_violation_ratio', np.nan)) else False:
        reasons.append('MAP_violation')
    if features.get('spike_ratio', 0) > 0.10:
        reasons.append('spikes')
    if features.get('saturation_ratio', 0) > 0.20:
        reasons.append('saturation')
    if features.get('missing_ratio', 0) > 0.30:
        reasons.append('missing_data')
    
    return reasons if reasons else ['none']


def process_case(caseid):
    """Process a single VitalDB case and extract all features"""
    
    try:
        # Load ABP waveform (5 minutes at 500 Hz = 150,000 samples)
        abp_wave = vitaldb.load_case(caseid, [TRACK_ABP_WAVE], 1/WAVEFORM_FREQ)
        
        if abp_wave is None:
            return None
        
        # Handle DataFrame vs array
        if hasattr(abp_wave, 'values'):
            abp_wave = abp_wave.values.flatten()
        else:
            abp_wave = np.asarray(abp_wave).flatten()
        
        # Take first 5 minutes
        n_samples = WINDOW_DURATION * WAVEFORM_FREQ
        if len(abp_wave) < n_samples:
            return None
        
        abp_wave = abp_wave[:n_samples]
        
        # Check for sufficient valid data
        valid_ratio = np.mean(~np.isnan(abp_wave))
        if valid_ratio < 0.5:
            return None
        
        # Load numeric vital signs
        numerics = vitaldb.load_case(caseid, TRACKS_NUMERICS, 1/NUMERIC_FREQ)
        
        if numerics is not None:
            if hasattr(numerics, 'values'):
                numerics = numerics.values
            else:
                numerics = np.asarray(numerics)
            
            n_numeric_samples = WINDOW_DURATION * NUMERIC_FREQ
            if len(numerics) >= n_numeric_samples:
                numerics = numerics[:n_numeric_samples]
                sbp = numerics[:, 0] if numerics.ndim > 1 else None
                dbp = numerics[:, 1] if numerics.ndim > 1 and numerics.shape[1] > 1 else None
                map_values = numerics[:, 2] if numerics.ndim > 1 and numerics.shape[1] > 2 else None
            else:
                sbp, dbp, map_values = None, None, None
        else:
            sbp, dbp, map_values = None, None, None
        
        # Extract all features
        features = {'caseid': caseid}
        features.update(extract_time_domain(abp_wave))
        features.update(extract_integrity(abp_wave))
        features.update(extract_physiologic(sbp, dbp, map_values))
        features.update(extract_alignment(abp_wave, sbp, dbp, map_values))
        
        # Generate label
        features['label'] = generate_label(features)
        features['failure_reasons'] = ','.join(get_failure_reasons(features))
        
        return features
        
    except Exception as e:
        return None


# ============================================================================
# MAIN EXTRACTION LOOP
# ============================================================================

print(f"\n[1/3] Starting extraction from VitalDB API...")
print("-"*70)

all_features = []
success_count = 0
fail_count = 0
start_time = time.time()

# Process cases
for caseid in tqdm(range(START_CASE, START_CASE + N_CASES_TARGET), desc="Processing cases"):
    features = process_case(caseid)
    
    if features is not None:
        all_features.append(features)
        success_count += 1
    else:
        fail_count += 1
    
    # Progress update every 50 cases
    if (caseid - START_CASE + 1) % 50 == 0:
        elapsed = time.time() - start_time
        rate = (caseid - START_CASE + 1) / elapsed
        remaining = (N_CASES_TARGET - (caseid - START_CASE + 1)) / rate if rate > 0 else 0
        print(f"\n  Progress: {caseid - START_CASE + 1}/{N_CASES_TARGET} | "
              f"Success: {success_count} | "
              f"ETA: {remaining/60:.1f} min")

elapsed_total = time.time() - start_time
print(f"\n[2/3] Extraction complete in {elapsed_total/60:.1f} minutes")
print(f"  Processed: {success_count + fail_count} cases")
print(f"  Successful: {success_count}")
print(f"  Failed/Skipped: {fail_count}")

# ============================================================================
# SAVE RESULTS
# ============================================================================

print(f"\n[3/3] Saving results...")

if len(all_features) > 0:
    df = pd.DataFrame(all_features)
    
    # Reorder columns
    feature_order = [
        'caseid',
        # Time-domain (6)
        'ABP_mean', 'ABP_std', 'ABP_min', 'ABP_max', 'ABP_range', 'ABP_IQR',
        # Physiologic (8)
        'PP_mean', 'PP_std', 'PP_violation_ratio', 'MAP_violation_ratio',
        'SBP_violation_ratio', 'DBP_violation_ratio', 'PP_negative_ratio', 'PP_extreme_ratio',
        # Integrity (6)
        'negative_value_ratio', 'zero_value_ratio', 'flatline_ratio', 
        'spike_ratio', 'saturation_ratio', 'missing_ratio',
        # Alignment (5)
        'r_MAP', 'MAD_MAP', 'r_SBP', 'r_DBP', 'alignment_score',
        # Labels
        'label', 'failure_reasons'
    ]
    
    # Only include columns that exist
    existing_cols = [c for c in feature_order if c in df.columns]
    df = df[existing_cols]
    
    # Save to CSV
    output_file = 'features_vitaldb.csv'
    df.to_csv(output_file, index=False)
    
    # Calculate statistics
    n_total = len(df)
    n_fail = (df['label'] == 'FAIL').sum()
    n_pass = (df['label'] == 'PASS').sum()
    fail_rate = 100 * n_fail / n_total
    
    print(f"\n{'='*70}")
    print("EXTRACTION RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"\n✓ Dataset saved: {output_file}")
    print(f"✓ Total windows: {n_total}")
    print(f"✓ PASS: {n_pass} ({100*n_pass/n_total:.1f}%)")
    print(f"✓ FAIL: {n_fail} ({fail_rate:.1f}%)")
    
    # Failure reasons breakdown
    print(f"\n✓ Failure Reasons Breakdown:")
    all_reasons = []
    for reasons in df['failure_reasons']:
        all_reasons.extend(reasons.split(','))
    
    from collections import Counter
    reason_counts = Counter(all_reasons)
    for reason, count in reason_counts.most_common():
        if reason != 'none':
            print(f"  - {reason}: {count} ({100*count/n_fail:.1f}% of failures)")
    
    # Key feature statistics
    print(f"\n✓ Key Feature Statistics:")
    print(f"  negative_value_ratio: {df['negative_value_ratio'].mean():.3f} ± {df['negative_value_ratio'].std():.3f}")
    print(f"  flatline_ratio:       {df['flatline_ratio'].mean():.3f} ± {df['flatline_ratio'].std():.3f}")
    print(f"  PP_violation_ratio:   {df['PP_violation_ratio'].mean():.3f} ± {df['PP_violation_ratio'].std():.3f}")
    
    print(f"\n{'='*70}")
    print("✓✓✓ EXTRACTION COMPLETE - REAL DATA FROM VITALDB API!")
    print(f"{'='*70}")
    
else:
    print("ERROR: No cases were successfully processed!")
