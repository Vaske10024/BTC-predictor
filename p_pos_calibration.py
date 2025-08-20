
import pandas as pd
from pathlib import Path
p = Path("paper_out_quick_relaxed2/trades_log_v7.csv")
df = pd.read_csv(p, parse_dates=['decision_time','entry_time','exit_time','signal_time'])
if 'future_return' not in df.columns:
    if 'exit_price' in df.columns and 'entry_price' in df.columns:
        df['future_return'] = (df['exit_price'] - df['entry_price']) / df['entry_price']
df['net_return_pct'] = df['future_return'] - 2*0.001 - 2*0.0005
# top losers
print("Top 15 losers (net_return_pct):")
print(df.nsmallest(15,'net_return_pct')[['decision_time','signal','p_pos','median_ret','future_return','net_return_pct']].to_string(index=False))
# group by signal
print("\nPer-signal stats:")
print(df.groupby('signal')['net_return_pct'].agg(['count','mean','median']).sort_values('count',ascending=False).to_string())

