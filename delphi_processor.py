#!/usr/bin/env python3
"""
Delphi Survey Data Processor Module
====================================

Core processing functions for Survey 1 → Survey 2 transformation.
Implements all aggregation, IQR, and peer list calculations per Delphi Methods spec.

Version: 2.0.0  —  Column names and ordering aligned to Qualtrics contact file template.

Key changes from v1:
  - S2_AGG column names: spaces stripped from measure tokens
      e.g.  S2_AGG_PGA_of_PsA_*   (not S2_AGG_PGA_of _PsA_*)
            S2_AGG_Serious_inf_*   (not S2_AGG_Serious _inf_*)
            S2_AGG_Upper_resp_inf_* (not S2_AGG_Upper_resp _inf_*)
  - IQR column names: mixed-case tokens preserving original casing
      e.g.  IQR_ARC20_*
            IQR_PGA_of_PsA_*
            IQR_Serious_inf_*
            IQR_Upper_resp_inf_*
            IQR_MDA_* / IQR_Nail_* / IQR_MACE_*
  - Output column sequence matches the Qualtrics contact file exactly:
      ResponseId, Email, [survey cols], social_desirability_*, S2_AGG_*, IQR_*, Peers_*
  - social_desirability_* columns are passed through from input unchanged
  - Unwanted input columns (Start Date, End Date, Unique Qualtrics ID,
    RecipientFirstName, RecipientLastName) are excluded from output
  - ResponseId is placed as the first column
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
MEASURE_FIELDS = [
    "Range_Lower","Range_Upper","Mode",
    "Quartile_1","Quartile_3","Median",
    "Mean","StdDev","Alpha","Beta"
]

T = 300        # Number of CDF sample points
EPS_MSE = 1e-20  # Minimum MSE to prevent division by zero

# Columns in the raw input that should NOT appear in the output.
# These are Qualtrics metadata columns that are never wanted downstream.
INPUT_COLS_TO_DROP = {
    "Start Date",
    "End Date",
    
}

# social_desirability columns to pass through (preserve order from input)
SOCIAL_DESIRABILITY_COLS = [
    "social_desirability_mode",
    "social_desirability_stddev",
    "social_desirability_min",
    "social_desirability_max",
    "social_desirability_alpha",
    "social_desirability_beta",
]

# ============================================================================
# TOKEN MAPPING  —  maps each measure to its column-name token
# ============================================================================
# Spaces are removed (replaced with nothing) in both S2_AGG and IQR tokens.
# The IQR tokens also remove the hyphen in ARC-20.

def _s2_agg_token(measure: str) -> str:
    return measure  # KEEP EXACT FORMAT


def _iqr_token(measure: str) -> str:
    """
    Column token used in IQR_* columns.
    Rule: strip all spaces AND the hyphen from the measure name.
    e.g.  'ARC-20'          -> 'ARC20'
          'PGA_of _PsA'     -> 'PGA_of_PsA'
          'Serious _inf'    -> 'Serious_inf'
          'Upper_resp _inf' -> 'Upper_resp_inf'
          'MDA'             -> 'MDA'
          'Nail'            -> 'Nail'
          'MACE'            -> 'MACE'
    """
    return measure.replace(" ", "").replace("-", "")


# ============================================================================
# COLUMN DEFINITIONS  —  ordered exactly as in the Qualtrics contact file
# ============================================================================

def get_required_columns() -> dict:
    s2_fields = MEASURE_FIELDS

    iqr_fields = [
        "Range_Lower_min", "Range_Lower_max",
        "Range_Upper_min", "Range_Upper_max",
        "Mode_min", "Mode_max",
        "StdDev_min", "StdDev_max"
    ]

    peer_fields = MEASURE_FIELDS

    columns = {'S2_AGG': [], 'IQR': [], 'Peers': []}

    # S2_AGG (KEEP ORIGINAL NAMES)
    for m in MEASURES:
        for f in s2_fields:
            columns['S2_AGG'].append(f"S2_AGG_{m}_{f}")

    # IQR (KEEP ORIGINAL NAMES)
    for m in MEASURES:
        for f in iqr_fields:
            columns['IQR'].append(f"IQR_{m}_{f}")

    # PEERS (KEEP ORIGINAL NAMES)
    for m in MEASURES:
        for f in peer_fields:
            columns['Peers'].append(f"Peers_{m}_{f}")

    return columns


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def normalize_token(s: str) -> str:
    """Remove non-alphanumeric characters and convert to lowercase (legacy use)."""
    return re.sub(r'[^0-9a-zA-Z]', '', s).lower()


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
    Per Delphi Methods specification section 5.3.
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return [np.nan] * len(probs)
    x = np.sort(x)
    n = len(x)
    out = []
    for p in probs:
        h = (n - 1) * p + 1
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
    """Calculate CDF of scaled Beta distribution on [min_val, max_val]."""
    if max_val == min_val:
        x_scaled = np.zeros_like(x)
        x_scaled[x >= max_val] = 1.0
        return x_scaled
    x_scaled = np.clip((x - min_val) / (max_val - min_val), 0, 1)
    return sp_beta.cdf(x_scaled, alpha, beta_param)


def beta_quantile(a: float, b: float, p: float, rmin: float, rmax: float) -> float:
    q = np.clip(sp_beta.ppf(p, a, b), 0, 1)
    return float(rmin + (rmax - rmin) * q)


def beta_mean(a: float, b: float, rmin: float, rmax: float) -> float:
    return float(rmin + (rmax - rmin) * (a / (a + b)))


def mode_sd_from_alpha_beta(alpha: float, beta_param: float,
                             rmin: float, rmax: float) -> Tuple[float, float]:
    """Calculate mode and SD from Beta parameters (Delphi spec §2.4)."""
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
    var_std = (alpha * beta_param) / ((alpha + beta_param) ** 2 * (alpha + beta_param + 1.0))
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
    """
    Compute weighted aggregate CDF using inverse-RMSE weighting.
    Per Delphi Methods §5.1.3.
    """
    N = len(peer_alpha)
    agg_min = float(np.min(peer_min))
    agg_max = float(np.max(peer_max))
    x_vals = np.linspace(agg_min, agg_max, T)

    cdf_mat = np.zeros((N, T))
    for j in range(N):
        cdf_mat[j, :] = cdf_beta_scaled(
            x_vals, peer_min[j], peer_max[j], peer_alpha[j], peer_beta[j]
        )

    mse = np.zeros(N)
    for j in range(N):
        others_mean = (cdf_mat[j, :] if N == 1
                       else np.mean(np.delete(cdf_mat, j, axis=0), axis=0))
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
    """
    Fit Beta distribution to aggregate CDF using L-BFGS-B.
    Per Delphi Methods §5.1.3.4.
    """
    x_scaled = np.clip((x_vals - agg_min) / (agg_max - agg_min), 0, 1)

    def objective(params):
        a, b = params
        if a <= 0 or b <= 0:
            return 1e12
        return np.sum((sp_beta.cdf(x_scaled, a, b) - target_cdf) ** 2)

    res = minimize(objective, x0=[2.0, 2.0],
                   bounds=[(0.1, 100.0), (0.1, 100.0)],
                   method='L-BFGS-B')
    return (float(res.x[0]), float(res.x[1])) if res.success else (2.0, 2.0)


def estimate_alpha_beta_from_mode_sd(
    range_min: float, range_max: float,
    mode_val: float, sd_val: float
) -> Tuple[Optional[float], Optional[float]]:
    """Estimate α and β from mode and SD (fallback when α/β not provided)."""
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
        var_std = (a * b) / ((a + b) ** 2 * (a + b + 1))
        return (pred_mode - mode_std) ** 2 + (np.sqrt(var_std) - sd_std) ** 2

    try:
        res = minimize(loss, x0=[2, 2], bounds=[(0.1, 100), (0.1, 100)])
        return (float(res.x[0]), float(res.x[1])) if res.success else (None, None)
    except Exception:
        return None, None


# ============================================================================
# INPUT VALIDATION
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

    found = [m for m in MEASURES
             if f"{m}_Range_Lower" in df.columns and f"{m}_Range_Upper" in df.columns]
    results['n_measures'] = len(found)

    if not found:
        results['is_valid'] = False
        results['errors'].append("No valid measures found in dataset")
    else:
        results['warnings'].append(f"Found {len(found)}/{len(MEASURES)} measures")

    if len(df) < 2:
        results['warnings'].append("Dataset has fewer than 2 respondents — peer comparisons limited")

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

    Output column order (mirrors Qualtrics contact file):
      1. ResponseId  (if present in input)
      2. Email       (if present in input)
      3. All survey measure columns  ({M}_Range_Lower … {M}_Beta  for each measure)
      4. Risk, Benefit  (if present)
      5. R_{measure}_* columns  (if present)
      6. social_desirability_* columns  (if present)
      7. S2_AGG_* columns
      8. IQR_* columns
      9. Peers_* columns

    Args:
        df:            Input dataframe with Survey 1 responses.
        handle_missing: Strategy for missing α/β (currently only default supported).
        verbose:       If True, log progress messages via callback.
        callback:      Optional callable(str) for progress updates.

    Returns:
        DataFrame with all generated columns added, ordered as above.
    """

    def log(msg: str):
        if verbose and callback:
            callback(msg)

    n = len(df)
    log(f"Processing {n} respondents…")

    # ── 1. Build the set of generated column names ────────────────────────────
    req = get_required_columns()
    float_cols = req['S2_AGG'] + req['IQR']
    peers_cols = req['Peers']

    # ── 2. Create output block (all NaN/empty, correct dtypes) ───────────────
    float_block = pd.DataFrame(np.nan, index=df.index, columns=float_cols, dtype=float)
    peers_block = pd.DataFrame("",     index=df.index, columns=peers_cols, dtype=object)

    # Concatenate input + generated columns
    working = pd.concat([df, float_block, peers_block], axis=1)

    log(f"Initialised {len(float_cols) + len(peers_cols)} output columns")

    # ── 3. Process each respondent ────────────────────────────────────────────
    for i in range(n):
        if callback and i % max(1, n // 10) == 0:
            callback(f"Processing respondent {i+1}/{n}")

        for m in MEASURES:
            src_min    = f"{m}_Range_Lower"
            src_max    = f"{m}_Range_Upper"
            src_mode   = f"{m}_Mode"
            src_sd     = f"{m}_StdDev"
            src_alpha  = f"{m}_Alpha"
            src_beta   = f"{m}_Beta"
            src_q1     = f"{m}_Quartile_1"
            src_q3     = f"{m}_Quartile_3"
            src_median = f"{m}_Median"
            src_mean   = f"{m}_Mean"

            if not (src_min in df.columns and src_max in df.columns):
                continue

            peer_mask = df.index != i
            peer_min = df.loc[peer_mask, src_min].astype(float).to_numpy()
            peer_max = df.loc[peer_mask, src_max].astype(float).to_numpy()

            valid = ~np.isnan(peer_min) & ~np.isnan(peer_max)
            if valid.sum() == 0:
                log(f"  Warning: No valid peer data for {m}, respondent {i+1}")
                continue

            peer_min  = peer_min[valid]
            peer_max  = peer_max[valid]
            peer_mode = (df.loc[peer_mask, src_mode].astype(float).to_numpy()[valid]
                         if src_mode in df.columns else np.full(len(peer_min), np.nan))
            peer_sd   = (df.loc[peer_mask, src_sd].astype(float).to_numpy()[valid]
                         if src_sd in df.columns else np.full(len(peer_min), np.nan))
            peer_q1   = (df.loc[peer_mask, src_q1].astype(float).to_numpy()[valid]
                         if src_q1 in df.columns else np.full(len(peer_min), np.nan))
            peer_q3   = (df.loc[peer_mask, src_q3].astype(float).to_numpy()[valid]
                         if src_q3 in df.columns else np.full(len(peer_min), np.nan))
            peer_median = (df.loc[peer_mask, src_median].astype(float).to_numpy()[valid]
                           if src_median in df.columns else np.full(len(peer_min), np.nan))
            peer_mean   = (df.loc[peer_mask, src_mean].astype(float).to_numpy()[valid]
                           if src_mean in df.columns else np.full(len(peer_min), np.nan))

            if src_alpha in df.columns and src_beta in df.columns:
                peer_alpha = df.loc[peer_mask, src_alpha].astype(float).to_numpy()[valid]
                peer_beta  = df.loc[peer_mask, src_beta].astype(float).to_numpy()[valid]
            else:
                peer_alpha = np.full(len(peer_min), np.nan)
                peer_beta  = np.full(len(peer_min), np.nan)

            for j in range(len(peer_alpha)):
                if np.isnan(peer_alpha[j]) or np.isnan(peer_beta[j]):
                    if not np.isnan(peer_mode[j]) and not np.isnan(peer_sd[j]):
                        est = estimate_alpha_beta_from_mode_sd(
                            peer_min[j], peer_max[j], peer_mode[j], peer_sd[j])
                        if est[0] is not None:
                            peer_alpha[j], peer_beta[j] = est

            if np.all(np.isnan(peer_alpha)):
                peer_alpha[:] = 2
                peer_beta[:]  = 2
            else:
                a_med = np.nanmedian(peer_alpha)
                b_med = np.nanmedian(peer_beta)
                peer_alpha = np.where(np.isnan(peer_alpha), a_med, peer_alpha)
                peer_beta  = np.where(np.isnan(peer_beta),  b_med, peer_beta)

            # ── S2_AGG ────────────────────────────────────────────────────────
            tok_s2 = _s2_agg_token(m)
            try:
                x_vals, agg_cdf, agg_min_v, agg_max_v = compute_aggregate_cdf_from_peers(
                    peer_min, peer_max, peer_alpha, peer_beta)
                agg_a, agg_b = fit_beta_to_cdf(x_vals, agg_cdf, agg_min_v, agg_max_v)
                agg_mode, agg_sd = mode_sd_from_alpha_beta(agg_a, agg_b, agg_min_v, agg_max_v)
                agg_q1     = beta_quantile(agg_a, agg_b, 0.25, agg_min_v, agg_max_v)
                agg_median = beta_quantile(agg_a, agg_b, 0.50, agg_min_v, agg_max_v)
                agg_q3     = beta_quantile(agg_a, agg_b, 0.75, agg_min_v, agg_max_v)
                agg_mean_v = beta_mean(agg_a, agg_b, agg_min_v, agg_max_v)

                working.at[i, f"S2_AGG_{tok_s2}_Range_Lower"] = agg_min_v
                working.at[i, f"S2_AGG_{tok_s2}_Range_Upper"] = agg_max_v
                working.at[i, f"S2_AGG_{tok_s2}_Mode"]        = agg_mode
                working.at[i, f"S2_AGG_{tok_s2}_Quartile_1"]  = agg_q1
                working.at[i, f"S2_AGG_{tok_s2}_Quartile_3"]  = agg_q3
                working.at[i, f"S2_AGG_{tok_s2}_Median"]      = agg_median
                working.at[i, f"S2_AGG_{tok_s2}_Mean"]        = agg_mean_v
                working.at[i, f"S2_AGG_{tok_s2}_StdDev"]      = agg_sd
                working.at[i, f"S2_AGG_{tok_s2}_Alpha"]       = agg_a
                working.at[i, f"S2_AGG_{tok_s2}_Beta"]        = agg_b
            except Exception as e:
                log(f"  Error computing S2_AGG for {m}, respondent {i+1}: {e}")

            # ── IQR ───────────────────────────────────────────────────────────
            tok_iqr   = _iqr_token(m)
            iqr_lower = r_quantile_type7(peer_min,  [0.25, 0.75])
            iqr_upper = r_quantile_type7(peer_max,  [0.25, 0.75])
            iqr_mode  = r_quantile_type7(peer_mode, [0.25, 0.75])
            iqr_sd    = r_quantile_type7(peer_sd,   [0.25, 0.75])

            working.at[i, f"IQR_{tok_iqr}_Range_Lower_min"] = iqr_lower[0]
            working.at[i, f"IQR_{tok_iqr}_Range_Lower_max"] = iqr_lower[1]
            working.at[i, f"IQR_{tok_iqr}_Range_Upper_min"] = iqr_upper[0]
            working.at[i, f"IQR_{tok_iqr}_Range_Upper_max"] = iqr_upper[1]
            working.at[i, f"IQR_{tok_iqr}_Mode_min"]        = iqr_mode[0]
            working.at[i, f"IQR_{tok_iqr}_Mode_max"]        = iqr_mode[1]
            working.at[i, f"IQR_{tok_iqr}_StdDev_min"]      = iqr_sd[0]
            working.at[i, f"IQR_{tok_iqr}_StdDev_max"]      = iqr_sd[1]

            # ── Peers ─────────────────────────────────────────────────────────
            working.at[i, f"Peers_{m}_Range_Lower"] = join_vals(peer_min)
            working.at[i, f"Peers_{m}_Range_Upper"] = join_vals(peer_max)
            working.at[i, f"Peers_{m}_Mode"]        = join_vals(peer_mode)
            working.at[i, f"Peers_{m}_Quartile_1"]  = join_vals(peer_q1)
            working.at[i, f"Peers_{m}_Quartile_3"]  = join_vals(peer_q3)
            working.at[i, f"Peers_{m}_Median"]      = join_vals(peer_median)
            working.at[i, f"Peers_{m}_Mean"]        = join_vals(peer_mean)
            working.at[i, f"Peers_{m}_StdDev"]      = join_vals(peer_sd)
            working.at[i, f"Peers_{m}_Alpha"]       = join_vals(peer_alpha)
            working.at[i, f"Peers_{m}_Beta"]        = join_vals(peer_beta)

    log("Processing complete!")

    # ── 4. Build final output with correct column order ───────────────────────
    return _reorder_output(working, req)


def _reorder_output(df: pd.DataFrame, req: dict) -> pd.DataFrame:
    """
    Reorder and filter output columns to match the Qualtrics contact file template:

      ResponseId, Email,
      [measure input cols in original order],
      [R_ cols if present],
      [Risk, Benefit if present],
      social_desirability_* (those present),
      S2_AGG_* (all 70),
      IQR_*    (all 56),
      Peers_*  (all 70)
    """
    all_cols = list(df.columns)

    # Columns we've generated (don't include in "input passthrough")
    generated = set(req['S2_AGG'] + req['IQR'] + req['Peers'])

    # Input columns to carry through (everything not generated and not dropped)
    input_passthrough = [c for c in all_cols
                         if c not in generated and c not in INPUT_COLS_TO_DROP]

    # Build final ordered list
    ordered: List[str] = []

    # ResponseId first (if present)
    if "ResponseId" in input_passthrough:
        ordered.append("ResponseId")
        input_passthrough.remove("ResponseId")

    # Email second (if present)
    if "Email" in input_passthrough:
        ordered.append("Email")
        input_passthrough.remove("Email")

    # Remaining input cols, but put social_desirability at end of input block
    social_present = [c for c in SOCIAL_DESIRABILITY_COLS if c in input_passthrough]
    non_social = [c for c in input_passthrough if c not in social_present]

    ordered.extend(non_social)
    ordered.extend(social_present)

    # Generated columns in spec order
    ordered.extend(req['S2_AGG'])
    ordered.extend(req['IQR'])
    ordered.extend(req['Peers'])

    # Keep only cols that actually exist in the dataframe
    ordered = [c for c in ordered if c in df.columns]

    return df[ordered]
