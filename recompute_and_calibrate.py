#!/usr/bin/env python3
"""
Recompute realized returns for each decision row from OHLCV and run calibration grid.
Usage:
python recompute_and_calibrate.py --trades_csv ./paper_out_quick_relaxed/trades_log_v7.csv --ohlcv_csv ./paper_out_quick_relaxed/ohlcv_used.csv --output_dir ./calib_out_recomputed --hold_grid 4 6 8 12 --min_trades 10
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd

def recompute_future_metrics(trades_df, ohlcv_df, hold):
    ohlcv = ohlcv_df.sort_index()
    trades = trades_df.copy().reset_index(drop=True)
    fut = []
    minf = []
    entry_prices = []
    exit_prices = []
    idxer = ohlcv.index.get_indexer

    for entry_time in trades['entry_time']:
        if pd.isna(entry_time):
            fut.append(np.nan); minf.append(np.nan); entry_prices.append(np.nan); exit_prices.append(np.nan)
            continue
        # find nearest ohlcv bar index
        pos = ohlcv.index.get_indexer([pd.to_datetime(entry_time)], method='nearest')[0]
        exit_pos = min(pos + int(hold), len(ohlcv)-1)
        entry_price = float(ohlcv.iloc[pos]['open'])
        exit_price = float(ohlcv.iloc[exit_pos]['close'])
        min_low = float(ohlcv.iloc[pos:exit_pos+1]['low'].min())
        fut.append((exit_price - entry_price) / entry_price)
        minf.append((min_low - entry_price) / entry_price)
        entry_prices.append(entry_price)
        exit_prices.append(exit_price)

    trades['future_return'] = fut
    trades['min_future_return'] = minf
    trades['entry_price_recomp'] = entry_prices
    trades['exit_price_recomp'] = exit_prices
    return trades

def evaluate_grid_on_df(df, prob_grid, thr_grid, atr_mult_grid, fee_pct=0.001, slippage_pct=0.0005, min_trades=5):
    results = []
    # ensure future_return present
    if 'future_return' not in df.columns:
        raise RuntimeError("df missing 'future_return' column")
    if 'atr_14' not in df.columns:
        df['atr_14'] = np.nan
    if 'min_future_return' not in df.columns:
        df['min_future_return'] = np.nan

    def compute_net_return_pct(row, atr_mult):
        entry = row.get('entry_price_recomp', row.get('entry_price', np.nan))
        fut_ret = row['future_return']
        if not np.isnan(row.get('min_future_return', np.nan)) and not np.isnan(row.get('atr_14', np.nan)):
            stop_trigger_ret = -atr_mult * row['atr_14'] / (entry if entry and not np.isnan(entry) else 1.0)
            if row['min_future_return'] <= stop_trigger_ret:
                realized_ret = stop_trigger_ret
            else:
                realized_ret = fut_ret
        else:
            realized_ret = fut_ret
        # subtract fees+slippage (entry+exit)
        net_ret = realized_ret - 2.0 * fee_pct - 2.0 * slippage_pct
        return net_ret

    for prob in prob_grid:
        for thr in thr_grid:
            for am in atr_mult_grid:
                cond = (df['p_pos'] >= prob) & (df['median_ret'] >= thr)
                cand = df[cond].copy()
                if len(cand) < min_trades:
                    continue
                net_returns = cand.apply(lambda r: compute_net_return_pct(r, am), axis=1).values
                net_returns = net_returns[~np.isnan(net_returns)]
                if len(net_returns) == 0:
                    continue
                avg_net_pct = float(net_returns.mean())
                total_net_pct = float(net_returns.sum())
                n_trades = len(net_returns)
                win_rate = float((net_returns > 0).sum() / n_trades)
                # bootstrap CI
                try:
                    boots = []
                    for _ in range(300):
                        s = np.random.choice(net_returns, size=n_trades, replace=True)
                        boots.append(np.mean(s))
                    ci_l, ci_u = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
                except Exception:
                    ci_l, ci_u = np.nan, np.nan
                results.append({
                    'prob': prob, 'thr': thr, 'atr_mult': am, 'n_trades': n_trades,
                    'avg_net_return_pct': avg_net_pct, 'total_net_return_pct': total_net_pct,
                    'win_rate': win_rate, 'avg_net_return_ci_l': ci_l, 'avg_net_return_ci_u': ci_u
                })
    return pd.DataFrame(results)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trades_csv", required=True)
    p.add_argument("--ohlcv_csv", required=True)
    p.add_argument("--output_dir", default="./calib_out_recomputed")
    p.add_argument("--hold_grid", nargs="+", type=int, default=[4,6,8,12])
    p.add_argument("--fee_pct", type=float, default=0.001)
    p.add_argument("--slippage_pct", type=float, default=0.0005)
    p.add_argument("--min_trades", type=int, default=10)
    args = p.parse_args()

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    trades = pd.read_csv(args.trades_csv, parse_dates=['decision_time','entry_time','exit_time','signal_time'])
    ohlcv = pd.read_csv(args.ohlcv_csv, parse_dates=['timestamp']).set_index('timestamp')
    all_results = []
    for hold in args.hold_grid:
        print("Recomputing metrics for hold =", hold)
        df_hold = recompute_future_metrics(trades, ohlcv, hold)
        # preserve atr_14 if present in original trades; otherwise, try compute simple atr from ohlcv
        if 'atr_14' not in df_hold.columns or df_hold['atr_14'].isna().all():
            # compute ATR on OHLCV and align to entry_time (approx)
            ohlcv_local = ohlcv.copy()
            ohlcv_local['tr'] = (ohlcv_local['high'] - ohlcv_local['low']).abs()
            ohlcv_local['atr_14'] = ohlcv_local['tr'].rolling(14, min_periods=1).mean()
            # map atr to trades by nearest timestamp
            idxs = ohlcv_local.index.get_indexer(df_hold['entry_time'].values, method='nearest')
            df_hold['atr_14'] = ohlcv_local['atr_14'].values[idxs]
        # run grid
        prob_grid = [0.9, 0.95]
        thr_grid = [0.01, 0.02, 0.03]  # require predicted move >= 1%, 2%, 3%
        atr_mult_grid = [1.5, 2.0, 3.0]

        df_res = evaluate_grid_on_df(df_hold, prob_grid, thr_grid, atr_mult_grid,
                                     fee_pct=args.fee_pct, slippage_pct=args.slippage_pct, min_trades=args.min_trades)
        if df_res is None or df_res.empty:
            continue
        df_res['hold'] = hold
        all_results.append(df_res)
    if not all_results:
        print("No results found for any hold.")
        return
    all_df = pd.concat(all_results, ignore_index=True)
    # sort by avg_net_return_pct then total_net_return_pct
    all_df = all_df.sort_values(['avg_net_return_pct','total_net_return_pct'], ascending=False).reset_index(drop=True)
    all_df.to_csv(Path(out_dir)/"calibration_grid_results.csv", index=False)
    best = all_df.iloc[0].to_dict()
    with open(Path(out_dir)/"calib_best.json","w") as fh:
        json.dump(best, fh, indent=2)
    print("Saved calibration results to:", out_dir)
    print("Top 10 configs:\n", all_df.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
