# calibrate_from_predictions.py
"""
Run a lightweight predictive pass and calibrate thresholds automatically.

Usage (PowerShell):
python calibrate_from_predictions.py --output_dir .\calib_out --max_bars 2000 --epochs 6 --num_samples 150

Notes:
- This is a fast calibration step to find reasonable threshold_pct & prob_threshold and stop multipliers.
- It will NOT do full retraining optimization. After calibration, use the recommended parameters in a full backtest.
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import ccxt
import time
import math
import warnings

# import the model/predict code path from your main script module
# we'll replicate minimal prediction logic here by importing relevant functions if needed.
# For simplicity, we'll re-use paper_nbeats_papertrade_gpu_patched_fixed_v7's walk_forward_pass
# If you don't have an importable module, this script will fetch OHLCV and do a light predictive pass
# by calling the same CLI script with reduced epochs and num_samples and reading its predictions CSV.
# To keep it standalone, this script will call the CLI script to produce 'predictions_only.csv'.
import subprocess
import sys

def run_predictive_pass(cli_script, args_list, out_csv):
    # Calls the v7 script with a special flag --predictions_only which we do not have,
    # so instead run the normal script but with --min_allocation 0 and a dedicated small output dir,
    # then use its trades_log_v7.csv where we extract decision-time predictions and realized returns.
    # Simpler: require that the user runs v7 once with relaxed flags and output_dir set.
    raise RuntimeError("Please run the main script once with relaxed flags and --output_dir set; then run this calibration pointing to that output_dir's trades_log_v7.csv")

def evaluate_grid(pred_df, fee_pct=0.001, slippage_pct=0.0005, min_trades=5):
    """
    Evaluate candidate threshold/prob/atr_mult/hold combos using a predictions DataFrame.

    Expects columns:
      - p_pos (float 0..1)
      - median_ret (float: return or predicted price delta depending on how preds were saved)
      - entry_price (float)
      - exit_price (float)  OR 'future_return' (exit/entry - 1)
      - atr_14 (float, in price units)
      - min_future_return (optional): (min_low - entry)/entry over hold period, used to detect early stop

    Returns a DataFrame of candidate configurations sorted by avg_net_return_pct desc.
    """

    # grids (tweak if you want)
    prob_grid = [0.7, 0.75, 0.8, 0.85, 0.9]
    thr_grid = [0.0005, 0.001, 0.0015, 0.002, 0.003]
    atr_mult_grid = [1.5, 2.0, 2.5, 3.0]
    hold_grid = [4, 6, 8, 12]

    results = []

    # normalize/ensure columns
    df = pred_df.copy()
    if 'future_return' not in df.columns:
        if 'exit_price' in df.columns and 'entry_price' in df.columns:
            df['future_return'] = (df['exit_price'] - df['entry_price']) / df['entry_price']
        else:
            raise RuntimeError("pred_df must contain either 'future_return' or both 'entry_price' and 'exit_price'")

    if 'atr_14' not in df.columns:
        df['atr_14'] = np.nan

    if 'min_future_return' not in df.columns:
        df['min_future_return'] = np.nan

    # helper to compute net return% per trade (entry->exit)
    def compute_net_return_pct(row, atr_mult):
        """
        return: net_return_pct (float)
        logic:
          - if min_future_return exists and <= -atr_mult * atr_14 / entry_price -> assume stop hit, return = stop_price - entry / entry
          - else return future_return
          - subtract fees (fee_pct applied to entry+exit notional) and slippage (slippage_pct applied similarly)
        """
        entry = row.get('entry_price', np.nan)
        fut_ret = row['future_return']  # (exit-entry)/entry
        # check for stop hit
        if not np.isnan(row.get('min_future_return', np.nan)) and not np.isnan(row.get('atr_14', np.nan)):
            stop_trigger_ret = -atr_mult * row['atr_14'] / entry  # expressed as return
            if row['min_future_return'] <= stop_trigger_ret:
                # stop hit: net return before fees = stop_trigger_ret
                realized_ret = stop_trigger_ret
            else:
                realized_ret = fut_ret
        else:
            realized_ret = fut_ret

        # fees: assume fee_pct applies at entry and exit
        # effective fee in return terms: approximately 2 * fee_pct (entry + exit), relative to notional
        net_ret = realized_ret - (2.0 * fee_pct) - (2.0 * slippage_pct)
        return net_ret

    for prob in prob_grid:
        for thr in thr_grid:
            for am in atr_mult_grid:
                for hold in hold_grid:
                    cond = (df['p_pos'] >= prob) & (df['median_ret'] >= thr)
                    cand = df[cond].copy()
                    if len(cand) < min_trades:
                        continue

                    # compute net return % per trade
                    net_returns = cand.apply(lambda r: compute_net_return_pct(r, am), axis=1).values
                    # filter out nan
                    net_returns = net_returns[~np.isnan(net_returns)]
                    if len(net_returns) == 0:
                        continue

                    avg_net_pct = float(np.mean(net_returns))
                    total_net_pct = float(np.sum(net_returns))
                    n_trades = len(net_returns)

                    # some extra diagnostics:
                    win_rate = float((net_returns > 0).sum() / n_trades)
                    # bootstrap 95% CI for mean
                    try:
                        boots = []
                        for _ in range(300):
                            s = np.random.choice(net_returns, size=n_trades, replace=True)
                            boots.append(np.mean(s))
                        ci_l, ci_u = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
                    except Exception:
                        ci_l, ci_u = np.nan, np.nan

                    results.append({
                        'prob': prob, 'thr': thr, 'atr_mult': am, 'hold': hold,
                        'n_trades': n_trades,
                        'avg_net_return_pct': avg_net_pct,
                        'total_net_return_pct': total_net_pct,
                        'win_rate': win_rate,
                        'avg_net_return_ci_l': ci_l,
                        'avg_net_return_ci_u': ci_u
                    })

    res_df = pd.DataFrame(results)
    if res_df.empty:
        print("No candidate combos found (try lowering grids/min_trades).")
        return None

    # rank by avg_net_return_pct then total_net_return_pct
    res_df = res_df.sort_values(['avg_net_return_pct', 'total_net_return_pct'], ascending=False).reset_index(drop=True)
    return res_df

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--predictions_csv", type=str, default=None, help="If you already have trades_log_v7.csv from a predictions-only run, point here.")
    p.add_argument("--output_dir", type=str, default="./calib_out")
    p.add_argument("--fee_pct", type=float, default=0.001,
                   help="Taker fee (fraction) to use in calibration (applied at entry+exit)")
    p.add_argument("--slippage_pct", type=float, default=0.0005, help="Slippage fraction applied at entry+exit")
    p.add_argument("--min_trades", type=int, default=5)

    args = p.parse_args()


    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.predictions_csv is None:
        print("Please run the main script first with relaxed flags (min_allocation=1, small epochs) and set --output_dir.")
        print("Then re-run this script with --predictions_csv pointing to that output_dir/trades_log_v7.csv")
        return

    pred_path = Path(args.predictions_csv)
    if not pred_path.exists():
        print("predictions CSV not found:", pred_path)
        return

    df = pd.read_csv(pred_path, parse_dates=['decision_time','entry_time','exit_time','signal_time'])
    # require columns p_pos, median_ret, entry_price and we need future price horizon
    # compute future_return over hold_bars if hold_bars present in CSV, else approximate using exit_price/entry_price
    if 'exit_price' in df.columns and 'entry_price' in df.columns:
        # For rows that had no execution, entry_price may be NaN; we'll compute realized returns from df of OHLCV externally.
        df['future_return'] = (df['exit_price'] - df['entry_price']) / df['entry_price']
    else:
        print("CSV missing entry/exit prices; this script expects a predictions CSV produced with entry/exit prices present.")
        return

    # get ATR at decision time if present
    if 'atr_14' not in df.columns:
        df['atr_14'] = np.nan

    # compute coarse min future return approximation if low is present (for stop simulation)
    if 'low' in df.columns:
        df['min_future_return'] = (df['low'] - df['entry_price']) / df['entry_price']
    else:
        df['min_future_return'] = np.nan

    # run the grid evaluator using CLI-specified fee/slippage/min_trades
    grid_df = evaluate_grid(df, fee_pct=args.fee_pct, slippage_pct=args.slippage_pct, min_trades=args.min_trades)

    if grid_df is None:
        print("No grid results.")
        return

    grid_df.to_csv(out_dir / "calibration_grid_results.csv", index=False)
    print("Top 10 candidate configs:\n", grid_df.head(10).to_string(index=False))
    # save best config
    best = grid_df.iloc[0].to_dict()
    with open(out_dir / "calib_best.json", "w") as fh:
        json.dump(best, fh, indent=2)

if __name__ == "__main__":
    main()
