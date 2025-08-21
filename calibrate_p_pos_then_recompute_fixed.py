#!/usr/bin/env python3
"""
Fixed calibration and backtesting flow:
 - Loads trades and OHLCV data.
 - Shows initial diagnostics (p_pos value counts, raw correlation).
 - Calibrates p_pos using a dynamic method.
 - Auto-selects a probability grid based on the new calibrated values.
 - Recomputes performance metrics across grids for different holding periods.
 - Saves all results to a CSV and the best configuration to a JSON file.

Usage:
  python calibrate_p_pos_then_recompute_fixed.py
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json
import warnings
from sklearn.isotonic import IsotonicRegression

# --- Configuration ---
TRADES_CSV = Path("paper_out_quick_relaxed2/trades_log_v7.csv")
OHLCV_CSV = Path("paper_out_quick_relaxed2/ohlcv_used.csv")
OUT_DIR = Path("calib_out_pcalib2_fixed")

USE_MEDIAN_RET_FILTER = False

# --- OPTIMIZATION STEP ---
# The grids have been narrowed to fine-tune the best parameters from the last run.
# Original best: hold=8, atr_mult=1.5
HOLD_GRID = [7, 8, 9, 10]
THR_GRID = [0.0005, 0.001, 0.002, 0.005]
ATR_MULT_GRID = [1.3, 1.4, 1.5, 1.6, 1.7]
# --- END OPTIMIZATION ---

FEE_PCT = 0.001
SLIPPAGE_PCT = 0.0005
MIN_TRADES = 8


# --- End Configuration ---

def calibrate_p_pos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calibrates the p_pos column based on observed outcomes.
    """
    df_out = df.copy()
    n_unique_probs = df_out['p_pos'].nunique()
    print(f"\nFound {n_unique_probs} unique p_pos values for calibration.")

    if n_unique_probs < 10:
        print("--> Using direct mapping for calibration (ideal for few unique probabilities).")
        calibration_map = df_out.groupby('p_pos')['future_return'].apply(lambda x: (x > 0).mean()).to_dict()
        df_out['p_pos_calib'] = df_out['p_pos'].map(calibration_map)

        print("\n--- Direct Calibration Map ---")
        for p_raw, p_calib in calibration_map.items():
            print(f"  Raw p_pos: {p_raw:<4.2f} -> Calibrated: {p_calib:.4f}")
        print("------------------------------")
    else:
        print("--> Using Isotonic Regression for calibration (ideal for many unique probabilities).")
        y_binary = (df_out['future_return'] > 0).astype(int)
        ir = IsotonicRegression(out_of_bounds='clip')
        df_out['p_pos_calib'] = ir.fit_transform(df_out['p_pos'], y_binary)

    return df_out


def recompute_future_metrics(trades_df, ohlcv_df, hold):
    """Recomputes future returns for a specific holding period."""
    ohlcv = ohlcv_df.sort_index()
    trades = trades_df.copy().reset_index(drop=True)
    fut, minf, entry_prices, exit_prices = [], [], [], []

    for entry_time in trades['entry_time']:
        if pd.isna(entry_time):
            fut.append(np.nan);
            minf.append(np.nan);
            entry_prices.append(np.nan);
            exit_prices.append(np.nan)
            continue

        pos = ohlcv.index.get_indexer([pd.to_datetime(entry_time)], method='nearest')[0]
        exit_pos = min(pos + int(hold), len(ohlcv) - 1)
        entry_price = float(ohlcv.iloc[pos]['open'])
        exit_price = float(ohlcv.iloc[exit_pos]['close'])
        min_low = float(ohlcv.iloc[pos:exit_pos + 1]['low'].min())

        fut.append((exit_price - entry_price) / entry_price)
        minf.append((min_low - entry_price) / entry_price)
        entry_prices.append(entry_price)
        exit_prices.append(exit_price)

    trades['future_return'] = fut
    trades['min_future_return'] = minf
    trades['entry_price_recomp'] = entry_prices
    trades['exit_price_recomp'] = exit_prices
    return trades


def evaluate_grid_on_df(df, prob_grid, thr_grid, atr_mult_grid, fee_pct, slippage_pct, min_trades,
                        use_median_ret_filter):
    """Evaluates all combinations of parameters on the given dataframe."""
    rows = []
    if 'future_return' not in df.columns or 'p_pos_calib' not in df.columns:
        return pd.DataFrame()

    df['atr_14'] = df.get('atr_14', np.nan)

    loop_thr_grid = thr_grid if use_median_ret_filter else [0.0]

    for prob in prob_grid:
        for thr in loop_thr_grid:
            for am in atr_mult_grid:
                if use_median_ret_filter:
                    cond = (df['p_pos_calib'] >= prob) & (df['median_ret'] >= thr)
                else:
                    cond = (df['p_pos_calib'] >= prob)

                cand = df[cond].copy()

                if len(cand) < min_trades:
                    continue

                net_returns = []
                for _, r in cand.iterrows():
                    entry = r.get('entry_price_recomp', r.get('entry_price', np.nan))
                    fut = r['future_return']

                    if not pd.isna(r.get('min_future_return')) and not pd.isna(r.get('atr_14')) and not pd.isna(
                            entry) and entry > 0:
                        stop_trigger_ret = -am * r['atr_14'] / entry
                        realized = stop_trigger_ret if r['min_future_return'] <= stop_trigger_ret else fut
                    else:
                        realized = fut

                    net_ret = realized - (2 * fee_pct) - (2 * slippage_pct)
                    net_returns.append(net_ret)

                if net_returns:
                    net_returns = np.array(net_returns)
                    rows.append({
                        'prob': prob, 'thr': thr if use_median_ret_filter else 'disabled', 'atr_mult': am,
                        'n_trades': len(net_returns),
                        'avg_net_return_pct': float(net_returns.mean()),
                        'total_net_return_pct': float(net_returns.sum()),
                        'win_rate': float((net_returns > 0).mean())
                    })
    return pd.DataFrame(rows)


def main():
    """Main execution function."""
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    trades = pd.read_csv(TRADES_CSV, parse_dates=['decision_time', 'entry_time', 'exit_time', 'signal_time'])
    ohlcv = pd.read_csv(OHLCV_CSV, parse_dates=['timestamp']).set_index('timestamp')

    if 'future_return' not in trades.columns and 'exit_price' in trades.columns:
        trades['future_return'] = (trades['exit_price'] - trades['entry_price']) / trades['entry_price']
    trades = trades.dropna(subset=['p_pos', 'future_return']).reset_index(drop=True)

    print("=== BASIC DIAGNOSTICS ===")
    if not USE_MEDIAN_RET_FILTER:
        print("!!! Median Return Filter is DISABLED for this run !!!")

    print(f"Total usable rows: {len(trades)}")
    print(f"Unique p_pos values: {trades['p_pos'].nunique()}")
    print("p_pos value counts:\n", trades['p_pos'].value_counts().to_string())
    print(f"corr(p_pos, future_return) = {trades['p_pos'].corr(trades['future_return'])}")

    trades = calibrate_p_pos(trades)

    print("\nCalibration summary (p_pos -> avg observed pos rate):")
    summary = trades.groupby('p_pos').agg(
        p_pos_calib_mean=('p_pos_calib', 'mean'),
        observed_win_rate=('future_return', lambda s: (s > 0).mean())
    ).reset_index()
    print(summary.to_string(index=False))
    print(f"\ncorr(p_pos_calib, future_return) = {trades['p_pos_calib'].corr(trades['future_return'])}")

    unique_calib = np.unique(np.round(trades['p_pos_calib'], 3))
    quantiles = np.quantile(trades['p_pos_calib'], [0.5, 0.6, 0.7, 0.8, 0.9])
    base_grid = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    combined = list(unique_calib) + list(quantiles) + base_grid
    prob_grid = sorted(list(set(round(p, 3) for p in combined if 0.0 <= p <= 1.0)), reverse=True)[:15]
    print("\nAuto-chosen prob_grid:", prob_grid)

    all_results = []
    for hold in HOLD_GRID:
        print(f"\nRecomputing for hold: {hold}")
        tr_recomputed = recompute_future_metrics(trades.copy(), ohlcv, hold)

        if 'atr_14' not in tr_recomputed.columns or tr_recomputed['atr_t_14'].isna().all():
            ohl = ohlcv.copy()
            tr = (ohl['high'] - ohl['low']).abs()
            ohl['atr_14'] = tr.rolling(14, min_periods=1).mean()
            idxs = ohl.index.get_indexer(tr_recomputed['entry_time'].values, method='nearest')
            tr_recomputed['atr_14'] = ohl['atr_14'].values[idxs]

        res = evaluate_grid_on_df(tr_recomputed, prob_grid, THR_GRID, ATR_MULT_GRID, FEE_PCT, SLIPPAGE_PCT, MIN_TRADES,
                                  USE_MEDIAN_RET_FILTER)
        if not res.empty:
            res['hold'] = hold
            all_results.append(res)

    if not all_results:
        print("\nNo candidate strategies found. Consider adjusting grid parameters or lowering MIN_TRADES.")
        return

    all_df = pd.concat(all_results, ignore_index=True)
    all_df = all_df.sort_values(by='avg_net_return_pct', ascending=False).reset_index(drop=True)

    out_csv = OUT_DIR / "results_pcalib_fine_tuned.csv"
    all_df.to_csv(out_csv, index=False, float_format='%.6f')

    out_json = OUT_DIR / "best_pcalib_config_fine_tuned.json"
    with open(out_json, "w") as f:
        json.dump(all_df.iloc[0].to_dict(), f, indent=2)

    print(f"\n✅ Saved results to: {out_csv}")
    print("Top 10 performing configurations:\n")
    print(all_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()