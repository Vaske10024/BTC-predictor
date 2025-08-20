
import pandas as pd
p = "paper_out_quick_relaxed2/trades_log_v7.csv"
df = pd.read_csv(p, parse_dates=['decision_time','entry_time','exit_time','signal_time'])
# ensure future_return exists (recomputed file usually has it)
if 'future_return' not in df.columns:
    if 'exit_price' in df.columns and 'entry_price' in df.columns:
        df['future_return'] = (df['exit_price'] - df['entry_price'])/df['entry_price']
print("unique p_pos count:", df['p_pos'].nunique())
print("p_pos value counts:\n", df['p_pos'].value_counts().to_string())
# correlation
print("\nCorr(p_pos, future_return):", df[['p_pos','future_return']].dropna().corr().iloc[0,1])
# show empirical prob of positive return by p_pos
df2 = df.dropna(subset=['p_pos','future_return']).copy()
print("\nObserved positive rate by p_pos:\n", df2.groupby('p_pos').apply(lambda g: (g['future_return']>0).mean()).reset_index(name='pos_rate').to_string(index=False))

