# inspect_trades_v7.py
import pandas as pd
import numpy as np
from pathlib import Path

# point this to the CSV from your output_dir
p = Path("./best_run_out_quick_relaxed/trades_log_v7.csv")  # change path for other run
if not p.exists():
    print("File not found:", p)
    raise SystemExit(1)

df = pd.read_csv(p, parse_dates=['decision_time','entry_time','exit_time','signal_time'])
print("Total rows:", len(df))

exec_df = df[df['side'] != 'NONE'].copy()
print("Executed trades:", len(exec_df))

if exec_df.empty:
    raise SystemExit("No executed trades found")

# Basic stats
print("\nBasic stats on executed trades:")
print(exec_df[['net_pnl','gross_pnl','fee_usd','slippage_usd','allocation_usd']].describe().T)

# Distribution of exit reasons
print("\nExit reasons:")
print(exec_df['exit_reason'].value_counts())

# Win rate
wins = (exec_df['net_pnl'] > 0).sum()
print(f"\nWin rate: {wins}/{len(exec_df)} = {wins/len(exec_df):.3f}")

# Group by signal label
print("\nBy signal label:")
print(exec_df.groupby('signal')['net_pnl'].agg(['count','sum','mean','median']).sort_values('count', ascending=False))

# Inspect p_pos distribution and how it correlates with net_pnl
exec_df['p_pos_bucket'] = pd.qcut(exec_df['p_pos'].fillna(0), q=5, duplicates='drop')
print("\nNet PnL by p_pos bucket:")
print(exec_df.groupby('p_pos_bucket')['net_pnl'].agg(['count','sum','mean']).sort_index())

# Top winners and losers
print("\nTop 10 losers:")
print(exec_df.nsmallest(10, 'net_pnl')[['decision_time','signal','p_pos','median_ret','entry_price','exit_price','net_pnl','exit_reason']].to_string(index=False))

print("\nTop 10 winners:")
print(exec_df.nlargest(10, 'net_pnl')[['decision_time','signal','p_pos','median_ret','entry_price','exit_price','net_pnl','exit_reason']].to_string(index=False))

# Are high p_pos trades actually better?
high_conf = exec_df[exec_df['p_pos'] >= 0.8]
low_conf = exec_df[exec_df['p_pos'] < 0.8]
print(f"\nHigh-conf trades >=0.8: {len(high_conf)}   mean net_pnl: {high_conf['net_pnl'].mean():.4f}")
print(f"Low-conf trades <0.8: {len(low_conf)}    mean net_pnl: {low_conf['net_pnl'].mean():.4f}")

# Save a compact diagnostic CSV
exec_df.to_csv(p.parent / "executed_trades_diagnostic.csv", index=False)
print("\nSaved executed_trades_diagnostic.csv")
