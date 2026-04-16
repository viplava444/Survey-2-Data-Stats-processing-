#!/usr/bin/env python3
"""
Delphi Survey Data Processor Module - CORRECTED NAMING
=======================================================

Core processing functions for Survey 1 → Survey 2 transformation.
CRITICAL: Uses EXACT column naming per specification.

Author: Data Engineering Team
Version: 2.1.0 - EXACT NAMING FIX
"""

import numpy as np
import pandas as pd
import re
from scipy.stats import beta as sp_beta
from scipy.optimize import minimize
from typing import List, Tuple, Optional, Callable

# ============================================================================
# CONFIGURATION
# ============================================================================

MEASURES = [
    "ARC-20",
    "PGA_of _PsA",
    "MDA",
    "Nail",
    "Serious _inf",
    "Upper_resp _inf",
    "MACE"
]

T = 300  # Number of CDF sample points
EPS_MSE = 1e-20  # Minimum MSE to prevent division by zero

# ============================================================================
# CRITICAL: EXACT TOKEN MAPPING
# ============================================================================

def get_s2_agg_token(measure: str) -> str:
    """
    Get S2_AGG token format.
    CRITICAL: No spaces after "of", "Serious", "Upper_resp"
    Keep dash for ARC-20
    """
    mapping = {
        "ARC-20": "ARC-20",
        "PGA_of _PsA": "PGA_of_PsA",         # NO space after "of"
        "MDA": "MDA",
        "Nail": "Nail",
        "Serious _inf": "Serious_inf",       # NO space after "Serious"
        "Upper_resp _inf": "Upper_resp_inf", # NO spaces
        "MACE": "MACE"
    }
    return mapping.get(measure, measure)

def get_iqr_token(measure: str) -> str:
    """
    Get IQR token format.
    CRITICAL: No dash for ARC-20 (becomes ARC20)
    No spaces, mixed case preserved
    """
    mapping = {
        "ARC-20": "ARC20",                   # NO dash!
        "PGA_of _PsA": "PGA_of_PsA",         # Mixed case
        "MDA": "MDA",
        "Nail": "Nail",
        "Serious _inf": "Serious_inf",       # NO space
        "Upper_resp _inf": "Upper_resp_inf", # NO spaces
        "MACE": "MACE"
    }
    return mapping.get(measure, measure)

def get_peers_token(measure: str) -> str:
    """
    Get Peers token format.
    CRITICAL: Keep ORIGINAL measure name (with spaces!)
    """
    return measure  # No changes - preserve original spacing!

# ============================================================================
# COLUMN DEFINITIONS WITH EXACT NAMING
# ============================================================================

def get_required_columns() -> dict:
    """Get all required column groups in EXACT order and format."""
    
    columns = {
        'S2_AGG': [],
        'IQR': [],
        'Peers': []
    }
    
    # S2_AGG columns
    s2_fields = [
        "Range_Lower", "Range_Upper", "Mode",
        "Quartile_1", "Quartile_3", "Median",
        "Mean", "StdDev", "Alpha", "Beta"
    ]
    
    for m in MEASURES:
        token = get_s2_agg_token(m)
        for field in s2_fields:
            columns['S2_AGG'].append(f"S2_AGG_{token}_{field}")
    
    # IQR columns
    iqr_fields = [
        "Range_Lower_min", "Range_Lower_max",
        "Range_Upper_min", "Range_Upper_max",
        "Mode_min", "Mode_max",
        "StdDev_min", "StdDev_max"
    ]
    
    for m in MEASURES:
        token = get_iqr_token(m)
        for field in iqr_fields:
            columns['IQR'].append(f"IQR_{token}_{field}")
    
    # Peers columns - ONLY for measures in input data
    peer_fields = [
        "Range_Lower", "Range_Upper", "Mode",
        "Quartile_1", "Quartile_3", "Median",
        "Mean", "StdDev", "Alpha", "Beta"
    ]
    
    for m in MEASURES:
        token = get_peers_token(m)  # Keep original spacing!
        for field in peer_fields:
            columns['Peers'].append(f"Peers_{token}_{field}")
    
    return columns

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def join_vals(arr) -> str:
    """Join array values as comma-separated string."""
    arr = np.asarray(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    return ",".join(map(str, arr.tolist())) if len(arr) > 0 else ""

# ============================================================================
# R TYPE 7 QUANTILE FUNCTION
# ============================================================================

def r_quantile_type7(x, probs: List[float]) -> List[float]:
    """
    Calculate quantiles using R's Type 7 method (Hyndman & Fan 1996).
    Per Delphi Methods specification.
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    
    if len(x) == 0:
        return [np.nan] * len(probs)
    
    x = np.sort(x)
    n = len(x)
    out = []
    
    for p in probs:
        h = (n - 1) * p + 1  # R Type 7 formula
        j = int(np.floor(h))
        gamma = h - j
        
        if j <= 0:
            out.append(float(x[0]))
        elif j >= n:
            out.append(float(x[-1]))
        else:
            out.append(float((1 - gamma) * x[j - 1] + gamma * x[j]))
    
    return out

# ============================================================================
# BETA DISTRIBUTION FUNCTIONS
# ============================================================================

def cdf_beta_scaled(x, min_val: float, max_val: float, alpha: float, beta_param: float):
    """Calculate CDF of Beta distribution scaled to [min_val, max_val]."""
    if max_val == min_val:
        x_scaled = np.zeros_like(x)
        x_scaled[x >= max_val] = 1.0
        return x_scaled
    
    x_scaled = (x - min_val) / (max_val - min_val)
    x_scaled = np.clip(x_scaled, 0, 1)
    return sp_beta.cdf(x_scaled, alpha, beta_param)

def beta_quantile(a: float, b: float, p: float, rmin: float, rmax: float) -> float:
    """Calculate quantile of Beta distribution on [rmin, rmax]."""
    q = sp_beta.ppf(p, a, b)
    q = np.clip(q, 0, 1)
    return float(rmin + (rmax - rmin) * q)

def beta_mean(a: float, b: float, rmin: float, rmax: float) -> float:
    """Calculate mean of Beta distribution on [rmin, rmax]."""
    return float(rmin + (rmax - rmin) * (a / (a + b)))

def mode_sd_from_alpha_beta(alpha: float, beta_param: float, rmin: float, rmax: float) -> Tuple[float, float]:
    """Calculate mode and SD from Beta parameters with edge cases."""
    width = rmax - rmin
    denom = alpha + beta_param - 2.0
    
    if alpha > 1 and beta_param > 1:
        mode_std = (alpha - 1.0) / denom
    elif alpha <= 1 and beta_param > 1:
        mode_std = 0.0
    elif alpha > 1 and beta_param <= 1:
        mode_std = 1.0
    else:
        mode_std = 0.5
    
    mode = rmin + mode_std * width
    var_std = (alpha * beta_param) / ((alpha + beta_param)**2 * (alpha + beta_param + 1.0))
    sd = np.sqrt(var_std) * width
    
    return float(mode), float(sd)

# ============================================================================
# AGGREGATION FUNCTIONS
# ============================================================================

def compute_aggregate_cdf_from_peers(
    peer_min: np.ndarray,
    peer_max: np.ndarray,
    peer_alpha: np.ndarray,
    peer_beta: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Compute weighted aggregate CDF using RMSE weighting."""
    N = len(peer_alpha)
    agg_min = float(np.min(peer_min))
    agg_max = float(np.max(peer_max))
    
    x_vals = np.linspace(agg_min, agg_max, T)
    
    cdf_mat = np.zeros((N, T))
    for j in range(N):
        cdf_mat[j, :] = cdf_beta_scaled(x_vals, peer_min[j], peer_max[j], 
                                        peer_alpha[j], peer_beta[j])
    
    mse = np.zeros(N)
    for j in range(N):
        if N == 1:
            others_mean = cdf_mat[j, :]
        else:
            others_mean = np.mean(np.delete(cdf_mat, j, axis=0), axis=0)
        mse[j] = np.mean((cdf_mat[j, :] - others_mean) ** 2)
    
    mse = np.maximum(mse, EPS_MSE)
    weights = 1.0 / np.sqrt(mse)
    weights /= np.sum(weights)
    agg_cdf = weights @ cdf_mat
    
    return x_vals, agg_cdf, agg_min, agg_max

def fit_beta_to_cdf(
    x_vals: np.ndarray,
    target_cdf: np.ndarray,
    agg_min: float,
    agg_max: float
) -> Tuple[float, float]:
    """Fit Beta distribution to aggregate CDF using L-BFGS-B."""
    x_scaled = (x_vals - agg_min) / (agg_max - agg_min)
    x_scaled = np.clip(x_scaled, 0, 1)
    
    def objective(params):
        a, b = params
        if a <= 0 or b <= 0:
            return 1e12
        pred = sp_beta.cdf(x_scaled, a, b)
        return np.sum((pred - target_cdf) ** 2)
    
    res = minimize(objective, x0=[2.0, 2.0], 
                   bounds=[(0.1, 100.0), (0.1, 100.0)], 
                   method='L-BFGS-B')
    
    if res.success:
        return float(res.x[0]), float(res.x[1])
    else:
        return 2.0, 2.0

def estimate_alpha_beta_from_mode_sd(
    range_min: float,
    range_max: float,
    mode_val: float,
    sd_val: float
) -> Tuple[Optional[float], Optional[float]]:
    """Estimate α and β from mode and SD (fallback)."""
    width = range_max - range_min
    if width <= 0:
        return None, None
    
    mode_std = (mode_val - range_min) / width
    sd_std = sd_val / width
    
    def loss(params):
        a, b = params
        if a <= 0 or b <= 0:
            return 1e9
        denom = a + b - 2
        if denom == 0:
            return 1e9
        pred_mode = (a - 1) / denom
        var_std = (a * b) / ((a + b)**2 * (a + b + 1))
        pred_sd = np.sqrt(var_std)
        return (pred_mode - mode_std)**2 + (pred_sd - sd_std)**2
    
    try:
        res = minimize(loss, x0=[2, 2], bounds=[(0.1, 100), (0.1, 100)])
        return (float(res.x[0]), float(res.x[1])) if res.success else (None, None)
    except:
        return None, None

# ============================================================================
# VALIDATION
# ============================================================================

def validate_input_data(df: pd.DataFrame) -> dict:
    """Validate input data structure."""
    results = {
        'is_valid': True,
        'warnings': [],
        'errors': [],
        'n_rows': len(df),
        'n_cols': len(df.columns),
        'n_measures': 0
    }
    
    if df.empty:
        results['is_valid'] = False
        results['errors'].append("Dataset is empty")
        return results
    
    found_measures = []
    for m in MEASURES:
        min_col = f"{m}_Range_Lower"
        max_col = f"{m}_Range_Upper"
        if min_col in df.columns and max_col in df.columns:
            found_measures.append(m)
    
    results['n_measures'] = len(found_measures)
    
    if len(found_measures) == 0:
        results['is_valid'] = False
        results['errors'].append("No valid measures found in dataset")
    else:
        results['warnings'].append(f"Found {len(found_measures)}/{len(MEASURES)} measures")
    
    if len(df) < 2:
        results['warnings'].append("Dataset has less than 2 respondents - peer comparisons limited")
    
    return results

# ============================================================================
# MAIN PROCESSING FUNCTION
# ============================================================================

def process_survey_data(
    df: pd.DataFrame,
    handle_missing: str = "Use default α=2, β=2",
    verbose: bool = False,
    callback: Optional[Callable] = None
) -> pd.DataFrame:
    """
    Process Survey 1 data to generate Survey 2 contact data.
    Uses EXACT column naming per specification.
    """
    def log(msg: str):
        if verbose and callback:
            callback(msg)
    
    n = len(df)
    log(f"Processing {n} respondents...")
    
    # Get required columns with EXACT naming
    required_cols = get_required_columns()
    
    # Create output dataframe with correct dtypes
    # S2_AGG and IQR: numeric (float)
    # Peers: string (object)
    out = df.copy()
    
    # Add S2_AGG columns (numeric)
    for col in required_cols['S2_AGG']:
        out[col] = np.nan
    
    # Add IQR columns (numeric)
    for col in required_cols['IQR']:
        out[col] = np.nan
    
    # Add Peers columns (string/object) - CRITICAL FIX
    for col in required_cols['Peers']:
        out[col] = pd.Series(dtype='object')  # String type!
    
    all_new_cols = len(required_cols['S2_AGG']) + len(required_cols['IQR']) + len(required_cols['Peers'])
    log(f"Created {all_new_cols} new columns with EXACT naming")
    
    # Process each respondent
    for i in range(n):
        if callback and i % max(1, n // 10) == 0:
            callback(f"Processing respondent {i+1}/{n}")
        
        for m in MEASURES:
            # Source column names (original format)
            src_min = f"{m}_Range_Lower"
            src_max = f"{m}_Range_Upper"
            src_mode = f"{m}_Mode"
            src_sd = f"{m}_StdDev"
            src_alpha = f"{m}_Alpha"
            src_beta = f"{m}_Beta"
            src_q1 = f"{m}_Quartile_1"
            src_q3 = f"{m}_Quartile_3"
            src_median = f"{m}_Median"
            src_mean = f"{m}_Mean"
            
            # Check if measure exists
            if not (src_min in df.columns and src_max in df.columns):
                continue
            
            # Get peer data
            peer_mask = df.index != i
            peer_min = df.loc[peer_mask, src_min].astype(float).to_numpy()
            peer_max = df.loc[peer_mask, src_max].astype(float).to_numpy()
            
            valid = ~np.isnan(peer_min) & ~np.isnan(peer_max)
            if valid.sum() == 0:
                log(f"  Warning: No valid peer data for {m}, respondent {i+1}")
                continue
            
            peer_min = peer_min[valid]
            peer_max = peer_max[valid]
            
            # Extract peer statistics
            peer_mode = df.loc[peer_mask, src_mode].astype(float).to_numpy()[valid] if src_mode in df.columns else np.full(len(peer_min), np.nan)
            peer_sd = df.loc[peer_mask, src_sd].astype(float).to_numpy()[valid] if src_sd in df.columns else np.full(len(peer_min), np.nan)
            peer_q1 = df.loc[peer_mask, src_q1].astype(float).to_numpy()[valid] if src_q1 in df.columns else np.full(len(peer_min), np.nan)
            peer_q3 = df.loc[peer_mask, src_q3].astype(float).to_numpy()[valid] if src_q3 in df.columns else np.full(len(peer_min), np.nan)
            peer_median = df.loc[peer_mask, src_median].astype(float).to_numpy()[valid] if src_median in df.columns else np.full(len(peer_min), np.nan)
            peer_mean = df.loc[peer_mask, src_mean].astype(float).to_numpy()[valid] if src_mean in df.columns else np.full(len(peer_min), np.nan)
            
            # Extract or estimate α and β
            if src_alpha in df.columns and src_beta in df.columns:
                peer_alpha = df.loc[peer_mask, src_alpha].astype(float).to_numpy()[valid]
                peer_beta = df.loc[peer_mask, src_beta].astype(float).to_numpy()[valid]
            else:
                peer_alpha = np.full(len(peer_min), np.nan)
                peer_beta = np.full(len(peer_min), np.nan)
            
            # Fill missing α and β
            for j in range(len(peer_alpha)):
                if np.isnan(peer_alpha[j]) or np.isnan(peer_beta[j]):
                    if not np.isnan(peer_mode[j]) and not np.isnan(peer_sd[j]):
                        est = estimate_alpha_beta_from_mode_sd(
                            peer_min[j], peer_max[j], peer_mode[j], peer_sd[j]
                        )
                        if est[0] is not None:
                            peer_alpha[j], peer_beta[j] = est
            
            # Fallback
            if np.all(np.isnan(peer_alpha)):
                peer_alpha[:] = 2
                peer_beta[:] = 2
            else:
                a_med = np.nanmedian(peer_alpha)
                b_med = np.nanmedian(peer_beta)
                peer_alpha = np.where(np.isnan(peer_alpha), a_med, peer_alpha)
                peer_beta = np.where(np.isnan(peer_beta), b_med, peer_beta)
            
            # ========================================
            # COMPUTE AGGREGATE (S2_AGG) - EXACT NAMING
            # ========================================
            
            try:
                x_vals, agg_cdf, agg_min, agg_max = compute_aggregate_cdf_from_peers(
                    peer_min, peer_max, peer_alpha, peer_beta
                )
                
                agg_alpha, agg_beta = fit_beta_to_cdf(x_vals, agg_cdf, agg_min, agg_max)
                
                agg_mode, agg_sd = mode_sd_from_alpha_beta(agg_alpha, agg_beta, agg_min, agg_max)
                agg_q1 = beta_quantile(agg_alpha, agg_beta, 0.25, agg_min, agg_max)
                agg_median = beta_quantile(agg_alpha, agg_beta, 0.5, agg_min, agg_max)
                agg_q3 = beta_quantile(agg_alpha, agg_beta, 0.75, agg_min, agg_max)
                agg_mean = beta_mean(agg_alpha, agg_beta, agg_min, agg_max)
                
                # Store S2_AGG values with EXACT naming
                token = get_s2_agg_token(m)
                out.at[i, f"S2_AGG_{token}_Range_Lower"] = agg_min
                out.at[i, f"S2_AGG_{token}_Range_Upper"] = agg_max
                out.at[i, f"S2_AGG_{token}_Mode"] = agg_mode
                out.at[i, f"S2_AGG_{token}_Quartile_1"] = agg_q1
                out.at[i, f"S2_AGG_{token}_Quartile_3"] = agg_q3
                out.at[i, f"S2_AGG_{token}_Median"] = agg_median
                out.at[i, f"S2_AGG_{token}_Mean"] = agg_mean
                out.at[i, f"S2_AGG_{token}_StdDev"] = agg_sd
                out.at[i, f"S2_AGG_{token}_Alpha"] = agg_alpha
                out.at[i, f"S2_AGG_{token}_Beta"] = agg_beta
                
            except Exception as e:
                log(f"  Error computing aggregate for {m}, respondent {i+1}: {e}")
            
            # ========================================
            # COMPUTE IQR - EXACT NAMING
            # ========================================
            
            token = get_iqr_token(m)
            
            iqr_lower = r_quantile_type7(peer_min, [0.25, 0.75])
            iqr_upper = r_quantile_type7(peer_max, [0.25, 0.75])
            iqr_mode = r_quantile_type7(peer_mode, [0.25, 0.75])
            iqr_sd = r_quantile_type7(peer_sd, [0.25, 0.75])
            
            out.at[i, f"IQR_{token}_Range_Lower_min"] = iqr_lower[0]
            out.at[i, f"IQR_{token}_Range_Lower_max"] = iqr_lower[1]
            out.at[i, f"IQR_{token}_Range_Upper_min"] = iqr_upper[0]
            out.at[i, f"IQR_{token}_Range_Upper_max"] = iqr_upper[1]
            out.at[i, f"IQR_{token}_Mode_min"] = iqr_mode[0]
            out.at[i, f"IQR_{token}_Mode_max"] = iqr_mode[1]
            out.at[i, f"IQR_{token}_StdDev_min"] = iqr_sd[0]
            out.at[i, f"IQR_{token}_StdDev_max"] = iqr_sd[1]
            
            # ========================================
            # STORE PEER LISTS - EXACT NAMING
            # ========================================
            
            token = get_peers_token(m)  # Keeps original spacing!
            
            out.at[i, f"Peers_{token}_Range_Lower"] = join_vals(peer_min)
            out.at[i, f"Peers_{token}_Range_Upper"] = join_vals(peer_max)
            out.at[i, f"Peers_{token}_Mode"] = join_vals(peer_mode)
            out.at[i, f"Peers_{token}_Quartile_1"] = join_vals(peer_q1)
            out.at[i, f"Peers_{token}_Quartile_3"] = join_vals(peer_q3)
            out.at[i, f"Peers_{token}_Median"] = join_vals(peer_median)
            out.at[i, f"Peers_{token}_Mean"] = join_vals(peer_mean)
            out.at[i, f"Peers_{token}_StdDev"] = join_vals(peer_sd)
            out.at[i, f"Peers_{token}_Alpha"] = join_vals(peer_alpha)
            out.at[i, f"Peers_{token}_Beta"] = join_vals(peer_beta)
    
    log("Processing complete with EXACT column naming!")
    return out
