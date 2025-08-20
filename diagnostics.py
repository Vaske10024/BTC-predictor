# diagnostics.py
import pandas as pd
from pathlib import Path
p = Path("./paper_out_quick_relaxed/trades_log_v7.csv")
df = pd.read_csv(p, parse_dates=['decision_time','entry_time','exit_time','signal_time'])
print("columns:", df.columns.tolist())
cols = ['p_pos','median_ret','entry_price','exit_price','future_return','atr_14','low','min_future_return','net_pnl','allocation_usd','side']
for c in cols:
    if c in df.columns:
        print(c, "non-null:", df[c].notna().sum())
    else:
        print(c, "MISSING")
print("\nSample rows:\n", df.head(8).to_string())
