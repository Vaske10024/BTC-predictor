#!/usr/bin/env python3
"""
Fixed calibration flow:
 - loads trades CSV
 - computes/ensures future_return
 - shows diagnostics (unique p_pos, correlation)
 - fits isotonic regression to map p_pos -> empirical P(future_return>0)
 - auto-selects prob_grid based on calibrated values (so no empty result)
 - recomputes future_return & min_future_return from OHLCV for multiple holds
 - evaluates candidate grids and writes CSV + best JSON

Usage:
  python calibrate_p_pos_then_recompute_fixed.py
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json
import warnings

try:
    from sklearn.isotonic import IsotonicRegression
except Exception:
    raise RuntimeError("scikit-learn required: pip install scikit-learn")

TRADES_CSV = Path("paper_out_quick_relaxed2/trades_log_v7.csv")
OHLCV_CSV = Path("paper_out_quick_relaxed2/ohlcv_used.csv")
OUT_DIR = Path("calib_out_pcalib2_fixed"); OUT_DIR.mkdir(parents=True, exist_ok=True)

# params (change if you want)
hold_grid = [4, 6, 8, 12]
fee_pct = 0.001
slippage_pct = 0.0005
min_trades = 8

def recompute_future_metrics(trades_df, ohlcv_df, hold):
    ohlcv = ohlcv_df.sort_index()
    trades = trades_df.copy().reset_index(drop=True)
    fut=[]; minf=[]; entry_prices=[]; exit_prices=[]
    for entry_time in trades['entry_time']:
        if pd.isna(entry_time):
            fut.append(np.nan); minf.append(np.nan); entry_prices.append(np.nan); exit_prices.append(np.nan)
            continue
        pos = ohlcv.index.get_indexer([pd.to_datetime(entry_time)], method='nearest')[0]
        exit_pos = min(pos + int(hold), len(ohlcv)-1)
        entry_price = float(ohlcv.iloc[pos]['open']); exit_price = float(ohlcv.iloc[exit_pos]['close'])
        min_low = float(ohlcv.iloc[pos:exit_pos+1]['low'].min())
        fut.append((exit_price - entry_price)/entry_price)
        minf.append((min_low - entry_price)/entry_price)
        entry_prices.append(entry_price); exit_prices.append(exit_price)
    trades['future_return'] = fut
    trades['min_future_return'] = minf
    trades['entry_price_recomp'] = entry_prices
    trades['exit_price_recomp'] = exit_prices
    return trades

def evaluate_grid_on_df(df, prob_grid, thr_grid, atr_mult_grid, fee_pct, slippage_pct, min_trades):
    rows = []
    if 'future_return' not in df.columns:
        return pd.DataFrame(rows)
    if 'atr_14' not in df.columns:
        df['atr_14'] = np.nan
    for prob in prob_grid:
        for thr in thr_grid:
            for am in atr_mult_grid:
                cond = (df['p_pos_calib'] >= prob) & (df['median_ret'] >= thr)
                cand = df[cond].copy()
                if len(cand) < min_trades:
                    continue
                net_returns = []
                for _, r in cand.iterrows():
                    entry = r.get('entry_price_recomp', r.get('entry_price', np.nan))
                    fut = r['future_return']
                    if not np.isnan(r.get('min_future_return', np.nan)) and not np.isnan(r.get('atr_14', np.nan)):
                        stop_trigger_ret = -am * r['atr_14'] / (entry if entry and not np.isnan(entry) else 1.0)
                        realized = stop_trigger_ret if r['min_future_return'] <= stop_trigger_ret else fut
                    else:
                        realized = fut
                    net_ret = realized - 2.0 * fee_pct - 2.0 * slippage_pct
                    net_returns.append(net_ret)
                net_returns = np.array(net_returns)
                rows.append({
                    'prob': prob, 'thr': thr, 'atr_mult': am,
                    'n_trades': int(len(net_returns)),
                    'avg_net_return_pct': float(net_returns.mean()),
                    'total_net_return_pct': float(net_returns.sum()),
                    'win_rate': float((net_returns>0).mean())
                })
    return pd.DataFrame(rows)

def main():
    warnings.filterwarnings("ignore")
    trades = pd.read_csv(TRADES_CSV, parse_dates=['decision_time','entry_time','exit_time','signal_time'])
    ohlcv = pd.read_csv(OHLCV_CSV, parse_dates=['timestamp']).set_index('timestamp')

    # ensure future_return exists (try recomputed columns first)
    if 'future_return' not in trades.columns:
        if 'exit_price' in trades.columns and 'entry_price' in trades.columns:
            trades['future_return'] = (trades['exit_price'] - trades['entry_price']) / trades['entry_price']
    trades = trades.dropna(subset=['p_pos','future_return'])  # only rows we can use

    print("=== BASIC DIAGNOSTICS ===")
    print("rows total:", len(trades))
    print("unique p_pos:", trades['p_pos'].nunique())
    print("p_pos value counts:\n", trades['p_pos'].value_counts().to_string())
    corr = trades[['p_pos','future_return']].corr().iloc[0,1]
    print("corr(p_pos, future_return) =", corr)

    # fit isotonic
    from sklearn.isotonic import IsotonicRegression
    y = (trades['future_return'] > 0).astype(float).values
    x = trades['p_pos'].values
    iso = IsotonicRegression(out_of_bounds='clip')
    x_out = iso.fit_transform(x, y)
    trades['p_pos_calib'] = x_out

    print("\nCalibration summary (p_pos -> avg observed pos rate):")
    summary = (
        trades.groupby('p_pos')[['p_pos_calib', 'future_return']]
        .agg({
            'p_pos_calib': 'mean',
            'future_return': lambda s: (s > 0).mean()
        })
        .reset_index()
    )

    print("\nCalibration summary (p_pos -> avg observed pos rate):\n", summary.to_string(index=False))

    # diagnostic corr after calibration
    print("\ncorr(p_pos_calib, future_return) =", trades[['p_pos_calib','future_return']].corr().iloc[0,1])

    # build prob_grid automatically from observed calibrated values + some standard thresholds
    unique_vals = np.unique(np.round(trades['p_pos_calib'], 3))
    # pick up to 6 representative thresholds: unique values plus common quantiles
    qvals = np.quantile(trades['p_pos_calib'].values, [0.5,0.6,0.7,0.8,0.9])
    candidate = sorted(set(list(unique_vals) + [float(x) for x in qvals] + [0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]))
    # keep only between 0 and 1
    prob_grid = [float(round(p,3)) for p in candidate if 0.0 <= p <= 1.0]
    # remove duplicates and sort descending (we want stricter first)
    prob_grid = sorted(list(dict.fromkeys(prob_grid)), reverse=True)[:12]

    print("\nAuto-chosen prob_grid (from calibrated p_pos):", prob_grid)

    # thr_grid and atr_mult_grid defaults (you can edit)
    thr_grid = [0.0005, 0.001, 0.002, 0.005]
    atr_mult_grid = [1.5, 2.0, 2.5]

    all_results = []
    for hold in hold_grid:
        print("Recomputing for hold:", hold)
        tr = recompute_future_metrics(trades.copy(), ohlcv, hold)
        # fill atr_14 if missing
        if 'atr_14' not in tr.columns or tr['atr_14'].isna().all():
            ohl = ohlcv.copy()
            ohl['tr'] = (ohl['high'] - ohl['low']).abs()
            ohl['atr_14'] = ohl['tr'].rolling(14, min_periods=1).mean()
            idxs = ohl.index.get_indexer(tr['entry_time'].values, method='nearest')
            tr['atr_14'] = ohl['atr_14'].values[idxs]
        # evaluate using p_pos_calib
        res = evaluate_grid_on_df(tr, prob_grid, thr_grid, atr_mult_grid, fee_pct, slippage_pct, min_trades)
        if not res.empty:
            res['hold'] = hold
            all_results.append(res)

    if not all_results:
        print("No candidate combos passed min_trades. Consider lowering min_trades or adjusting grids.")
        return

    all_df = pd.concat(all_results, ignore_index=True)
    all_df = all_df.sort_values(['avg_net_return_pct','total_net_return_pct'], ascending=False).reset_index(drop=True)
    out_csv = OUT_DIR/"results_pcalib_fixed.csv"
    all_df.to_csv(out_csv, index=False)
    with open(OUT_DIR/"best_pcalib_config.json","w") as fh:
        json.dump(all_df.iloc[0].to_dict(), fh, indent=2)
    print("\nSaved results to:", out_csv)
    print("Top 10 configs:\n", all_df.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
