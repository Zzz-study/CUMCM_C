# -*- coding: utf-8 -*-
"""
Bayesian Beta-Binomial GLMM for success/trial data with GA (and optional BMI) and subject-level random intercepts.

Requirements (install once):
  pip install -U pymc arviz numpy pandas openpyxl

Input data requirement:
  A dataframe (or CSV/Excel) with at least columns:
    - subject_id : identifier per subject (string/int)
    - GA         : gestational age (numeric)
    - successes  : number of successes per observation (int >= 0)
    - trials     : number of trials per observation (int > 0)
    - BMI        : optional (numeric)

Outputs:
  - bbglmm_pred_curve.xlsx : Posterior mean/median and 95% CI of p(GA) on a GA grid
  - bbglmm_summary.xlsx    : Posterior summary for key parameters
  - Console print of earliest GA where median p >= threshold (default 4%)
"""

import warnings
warnings.filterwarnings('ignore')

import sys
import os
import argparse
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az


def fit_bbglmm(df_sample: pd.DataFrame,
               draws: int = 1500,
               tune: int = 1000,
               chains: int = 4,
               target_accept: float = 0.90,
               random_seed: int = 42,
               pred_thr: float = 0.04):
    """
    Fit Beta-Binomial GLMM with:
      - Fixed effects: intercept, GA (z-scored), GA^2, optional BMI (z-scored)
      - Random intercept: per subject
      - Likelihood: Beta-Binomial(successes | trials, alpha=p*phi, beta=(1-p)*phi)

    Returns:
      idata   : InferenceData
      pred    : DataFrame with GA grid and posterior summaries of p
      earliest: float or None (earliest GA where median p >= pred_thr)
      summary : ArviZ summary for key parameters
    """
    df = df_sample.copy()

    # Only keep required columns; BMI is optional
    has_bmi = ('BMI' in df.columns) and pd.to_numeric(df['BMI'], errors='coerce').notna().any()
    needed = ['subject_id', 'GA', 'successes', 'trials'] + (['BMI'] if has_bmi else [])
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[needed].copy()

    # Coerce numerics
    df['GA'] = pd.to_numeric(df['GA'], errors='coerce')
    df['successes'] = pd.to_numeric(df['successes'], errors='coerce')
    df['trials'] = pd.to_numeric(df['trials'], errors='coerce')
    if has_bmi:
        df['BMI'] = pd.to_numeric(df['BMI'], errors='coerce')

    # Basic cleaning
    df = df.dropna(subset=['GA', 'successes', 'trials'])
    df = df[(df['trials'] > 0) & (df['successes'] >= 0) & (df['successes'] <= df['trials'])].copy()

    # z-score for stability
    ga_mu, ga_sd = df['GA'].mean(), df['GA'].std(ddof=0)
    df['GA_z'] = (df['GA'] - ga_mu) / (ga_sd if ga_sd > 0 else 1.0)

    if has_bmi:
        bmi_mu, bmi_sd = df['BMI'].mean(), df['BMI'].std(ddof=0)
        df['BMI_z'] = (df['BMI'] - bmi_mu) / (bmi_sd if bmi_sd > 0 else 1.0)

    # Encode subjects
    subj_codes = pd.Categorical(df['subject_id'])
    subj_idx = subj_codes.codes.astype('int64')
    J = subj_codes.categories.size

    # Arrays
    n = df['trials'].astype('int64').values
    y = df['successes'].astype('int64').values
    ga_z = df['GA_z'].astype(float).values
    if has_bmi:
        bmi_z = df['BMI_z'].astype(float).values

    with pm.Model() as model:
        # Fixed effects
        beta0   = pm.Normal('beta0', 0, 2.5)
        beta_ga = pm.Normal('beta_ga', 0, 2.5)
        beta_ga2= pm.Normal('beta_ga2', 0, 2.5)
        if has_bmi:
            beta_bmi = pm.Normal('beta_bmi', 0, 2.5)

        # Random intercepts
        sigma_u = pm.HalfNormal('sigma_u', 1.0)
        u_raw   = pm.Normal('u_raw', 0, 1, shape=J)
        u       = pm.Deterministic('u', u_raw * sigma_u)

        # Linear predictor
        eta = beta0 + beta_ga * ga_z + beta_ga2 * (ga_z ** 2) + u[subj_idx]
        if has_bmi:
            eta = eta + beta_bmi * bmi_z

        p = pm.Deterministic('p', pm.math.sigmoid(eta))

        # Beta-binomial concentration (phi>0)
        phi = pm.HalfNormal('phi', 5.0)
        alpha = p * phi
        beta = (1.0 - p) * phi

        # Likelihood
        y_obs = pm.BetaBinomial('y_obs', n=n, alpha=alpha, beta=beta, observed=y)

        idata = pm.sample(draws=draws, tune=tune, chains=chains,
                          target_accept=target_accept, random_seed=random_seed,
                          progressbar=True)

    # Posterior predictive (population-average; u=0)
    ga_grid = np.arange(10, 28, 0.5)  # adjust as needed
    ga_grid_z = (ga_grid - ga_mu) / (ga_sd if ga_sd > 0 else 1.0)

    post = idata.posterior
    b0   = post['beta0'  ].stack(sample=('chain','draw')).values
    bga  = post['beta_ga'].stack(sample=('chain','draw')).values
    bga2 = post['beta_ga2'].stack(sample=('chain','draw')).values
    if has_bmi:
        bbmi = post['beta_bmi'].stack(sample=('chain','draw')).values
        bmi_z_grid = np.zeros_like(ga_grid)  # median BMI → z=0
    else:
        bmi_z_grid = None

    eta_grid = b0[:, None] + bga[:, None] * ga_grid_z[None, :] + bga2[:, None] * (ga_grid_z[None, :] ** 2)
    if has_bmi:
        eta_grid += bbmi[:, None] * bmi_z_grid[None, :]
    p_grid = 1.0 / (1.0 + np.exp(-eta_grid))

    p_mean = p_grid.mean(axis=0)
    p_med  = np.median(p_grid, axis=0)
    p_lo   = np.quantile(p_grid, 0.025, axis=0)
    p_hi   = np.quantile(p_grid, 0.975, axis=0)

    pred = pd.DataFrame({'GA': ga_grid, 'p_mean': p_mean, 'p_median': p_med, 'p_lo': p_lo, 'p_hi': p_hi})

    # Earliest GA where median p >= pred_thr
    hit = pred.loc[pred['p_median'] >= pred_thr]
    earliest = float(hit.iloc[0]['GA']) if len(hit) else None

    # Posterior summary
    var_names = ['beta0', 'beta_ga', 'beta_ga2', 'sigma_u', 'phi']
    if has_bmi:
        var_names.insert(3, 'beta_bmi')
    summary = az.summary(idata, var_names=var_names, kind='stats')

    return idata, pred, earliest, summary


def _read_table_auto(path: str, sheet: str = None) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in ['.xlsx', '.xls']:
        return pd.read_excel(path, sheet_name=sheet) if sheet else pd.read_excel(path)
    elif ext in ['.csv', '.txt']:
        return pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")


def main():
    parser = argparse.ArgumentParser(description="Fit Beta-Binomial GLMM with GA (and optional BMI) and subject random effects.")
    parser.add_argument('--input', type=str, required=False, help='Path to CSV/Excel with columns: subject_id, GA, successes, trials, optional BMI')
    parser.add_argument('--sheet', type=str, default=None, help='Excel sheet name (if input is Excel)')
    parser.add_argument('--draws', type=int, default=1500)
    parser.add_argument('--tune', type=int, default=1000)
    parser.add_argument('--chains', type=int, default=4)
    parser.add_argument('--target_accept', type=float, default=0.90)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--thr', type=float, default=0.04, help='Threshold on median p(GA) to report earliest GA')
    parser.add_argument('--out_pred', type=str, default='bbglmm_pred_curve.xlsx')
    parser.add_argument('--out_sum', type=str, default='bbglmm_summary.xlsx')
    args = parser.parse_args()

    if args.input:
        df_sample = _read_table_auto(args.input, sheet=args.sheet)
    else:
        print("No --input provided. Expect a dataframe with columns: subject_id, GA, successes, trials[, BMI].")
        print("Exiting without fitting.")
        sys.exit(0)

    idata, pred, earliest, summary = fit_bbglmm(
        df_sample=df_sample,
        draws=args.draws,
        tune=args.tune,
        chains=args.chains,
        target_accept=args.target_accept,
        random_seed=args.seed,
        pred_thr=args.thr
    )

    # Save outputs
    pred.to_excel(args.out_pred, index=False)
    summary_rounded = summary.round(3).reset_index().rename(columns={'index': 'param'})
    summary_rounded.to_excel(args.out_sum, index=False)

    print(f"✓ Saved: {args.out_pred}, {args.out_sum}")
    print(f"Earliest GA with median p >= {args.thr:.3f}: {earliest}")


if __name__ == '__main__':
    main()