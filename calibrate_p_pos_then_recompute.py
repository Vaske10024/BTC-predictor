# calibrate_p_pos_then_recompute.py
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.isotonic import IsotonicRegression
import json

# adjust paths
trades_csv = Path("paper_out_quick_relaxed2/trades_log_v7.csv")
ohlcv_csv = Path("paper_out_quick_relaxed2/ohlcv_used.csv")
out_dir = Path("calib_out_pcalib2"); out_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(trades_csv, parse_dates=['decision_time','entry_time','exit_time','signal_time'])
# recompute future_return if recomputed columns exist (recompute_and_calibrate already did that),
# here assume recomputed file has 'future_return'. If not, use exit/entry
if 'future_return' not in df.columns:
    if 'entry_price' in df.columns and 'exit_price' in df.columns:
        df['future_return'] = (df['exit_price'] - df['entry_price']) / df['entry_price']
    else:
        raise RuntimeError("No future_return / entry+exit prices found")

# build binary label: positive return
df = df.dropna(subset=['p_pos','future_return'])
y = (df['future_return'] > 0).astype(float).values
x = df['p_pos'].values

# Fit isotonic regression (monotonic non-decreasing mapping)
iso = IsotonicRegression(out_of_bounds='clip')
iso_y = iso.fit_transform(x, y)

df['p_pos_calib'] = iso_y

# inspect mapping summaries
mapping_df = pd.DataFrame({"p_pos": x, "calib": iso_y, "y": y})
print("Sample calibration mapping (unique p_pos -> avg observed prob):")
print(mapping_df.groupby('p_pos')['y'].mean().reset_index().head(20).to_string(index=False))

# save calibrated CSV
calib_csv = out_dir / "trades_with_p_pos_calib.csv"
df.to_csv(calib_csv, index=False)
print("Wrote:", calib_csv)

# Now run a small evaluation loop akin to recompute_and_calibrate but using p_pos_calib as filter
from pathlib import Path
ohlcv = pd.read_csv(ohlcv_csv, parse_dates=['timestamp']).set_index('timestamp')

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
    trades['future_return'] = fut; trades['min_future_return']=minf
    trades['entry_price_recomp']=entry_prices; trades['exit_price_recomp']=exit_prices
    return trades

# grids
prob_grid = [0.8, 0.85, 0.9, 0.95]   # now use p_pos_calib
thr_grid = [0.001, 0.002, 0.005]
atr_mult_grid = [1.5, 2.0, 3.0]
hold_grid = [4,6,8,12]
fee_pct = 0.001; slippage_pct = 0.0005; min_trades = 8

all_res=[]

for hold in hold_grid:
    tr = recompute_future_metrics(df.copy(), ohlcv, hold)
    # compute atr_14 from ohlcv if missing
    if 'atr_14' not in tr.columns or tr['atr_14'].isna().all():
        ohl = ohlcv.copy()
        ohl['tr'] = (ohl['high'] - ohl['low']).abs()
        ohl['atr_14'] = ohl['tr'].rolling(14, min_periods=1).mean()
        idxs = ohl.index.get_indexer(tr['entry_time'].values, method='nearest')
        tr['atr_14'] = ohl['atr_14'].values[idxs]
    for prob in prob_grid:
        for thr in thr_grid:
            for am in atr_mult_grid:
                cond = (tr['p_pos_calib'] >= prob) & (tr['median_ret'] >= thr)
                cand = tr[cond].copy()
                if len(cand) < min_trades:
                    continue
                # compute net returns
                net = []
                for _, r in cand.iterrows():
                    entry = r['entry_price_recomp']
                    fut = r['future_return']
                    if not np.isnan(r['min_future_return']) and not np.isnan(r['atr_14']):
                        stop_trigger_ret = -am * r['atr_14'] / (entry if entry else 1.0)
                        realized = stop_trigger_ret if r['min_future_return'] <= stop_trigger_ret else fut
                    else:
                        realized = fut
                    net_ret = realized - 2*fee_pct - 2*slippage_pct
                    net.append(net_ret)
                net = np.array(net)
                all_res.append({
                    'hold': hold, 'prob': prob, 'thr': thr, 'atr_mult': am,
                    'n_trades': len(net), 'avg_net_return_pct': net.mean(), 'total_net_return_pct': net.sum(),
                    'win_rate': (net>0).mean()
                })

res_df = pd.DataFrame(all_res).sort_values(['avg_net_return_pct','total_net_return_pct'], ascending=False)
res_df.to_csv(out_dir/"results_pcalib.csv", index=False)
print("Saved results to", out_dir/"results_pcalib.csv")
print(res_df.head(15).to_string(index=False))
