#!/usr/bin/env python3
"""
paper_nbeats_papertrade_gpu_patched.py

Patched version with:
 - GPU autodetection for PyTorch / Darts (NVIDIA GTX 1660 Super should work if you have CUDA + proper torch)
 - Automatic training on returns when possible
 - Probability gating (p_pos) using samples to avoid tiny/noisy trades
 - Smaller default position sizing
 - torch.set_num_threads support via CLI
 - Buy-and-hold baseline printed
 - Some defensive checks and helpful prints

Notes:
 - Make sure you installed a CUDA-enabled PyTorch matching your GPU (e.g. cuda11x wheel from pytorch.org).
 - If you don't have GPU PyTorch, the script will run on CPU automatically.
"""

from darts.utils.likelihood_models.torch import QuantileRegression

import argparse
import time
from pathlib import Path
import math
import warnings

import ccxt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# Darts
from darts import TimeSeries
from darts.models import NBEATSModel

# torch for device control
import torch

warnings.filterwarnings("ignore")

# ---------------------
# Helpers: metrics, fetcher, trading primitives
# ---------------------

def fetch_ohlcv_full(exchange, symbol="BTC/USDT", timeframe="1h", since=None, limit=1000, max_bars=3000):
    all_klines = []
    ex = exchange
    fetch_since = since
    pbar = tqdm(total=max_bars, desc=f"Fetching {symbol} {timeframe}")
    while True:
        try:
            chunk = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=fetch_since, limit=limit)
        except Exception as e:
            print("fetch error, sleeping 1s:", e)
            time.sleep(1)
            continue

        if not chunk:
            break

        all_klines.extend(chunk)
        pbar.update(len(chunk))

        last_ts = chunk[-1][0]
        fetch_since = last_ts + 1
        if len(chunk) < limit or len(all_klines) >= max_bars:
            break
        time.sleep(0.12)

    pbar.close()
    if not all_klines:
        raise RuntimeError("No OHLCV returned")

    df = pd.DataFrame(all_klines, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.set_index("timestamp")
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def max_drawdown(equity):
    equity = np.asarray(equity)
    highwater = np.maximum.accumulate(equity)
    dd = (equity - highwater) / highwater
    return float(np.min(dd))


def annualized_sharpe(returns, periods_per_year):
    r = np.array(returns)
    if r.size < 2:
        return 0.0
    mean_r = np.mean(r)
    std_r = np.std(r, ddof=1)
    if std_r == 0:
        return 0.0
    return float((mean_r / std_r) * math.sqrt(periods_per_year))


# default position sizing mapping (fraction of equity) -- reduced defaults
DEFAULT_POS_SIZING = {
    "STRONG_BUY": 0.12,
    "BUY": 0.06,
    "HOLD": 0.0,
    "SELL": -0.06,
    "STRONG_SELL": -0.12,
}


# Decision mapping from quantiles -> signal (kept for fallback)
def decide_signal_from_quantiles(median_ret, low_ret, high_ret, thr):
    if median_ret >= thr and low_ret > 0:
        return "STRONG_BUY"
    if median_ret >= thr and low_ret <= 0:
        return "BUY"
    if abs(median_ret) < thr:
        return "HOLD"
    if median_ret <= -thr and high_ret < 0:
        return "STRONG_SELL"
    if median_ret <= -thr and high_ret >= 0:
        return "SELL"
    return "HOLD"


# ---------------------
# Core backtest / live-sim routine
# ---------------------

def extract_samples_from_prob_pred(prob_pred):
    """Try several ways to extract raw samples array from a Darts probabilistic prediction.
    Returns numpy array shaped (samples, length) or raises RuntimeError.
    """
    # common API in newer versions: .all_values()
    getter = getattr(prob_pred, "all_values", None)
    if callable(getter):
        arr = getter()
        return np.asarray(arr)

    # fallback: samples property
    getter = getattr(prob_pred, "samples", None)
    if callable(getter):
        arr = getter()
        return np.asarray(arr)

    # try attribute directly
    arr = getattr(prob_pred, "_all_values", None)
    if arr is not None:
        return np.asarray(arr)

    raise RuntimeError("Cannot extract samples from probabilistic prediction (no all_values/samples API)")


def walk_forward_backtest(df, args):
    close = df["close"]
    n = len(df)
    train_size = int(args.train_frac * n)
    if train_size < 100:
        raise ValueError("Train fraction too small or not enough bars; increase max_bars or train_frac")

    train_end_idx = train_size
    test_start_idx = train_end_idx
    test_end_idx = n - args.hold_bars  # need hold_bars for exit
    print(f"Total bars: {n}, train_end_idx={train_end_idx}, test from {test_start_idx} to {test_end_idx-1}")

    capital = float(args.initial_capital)
    equity_ts = []
    equity_time = []
    trades = []

    # estimate periods_per_year from timeframe
    tf = args.timeframe
    if tf.endswith("h"):
        hours = int(tf[:-1])
        periods_per_year = int(24 / hours * 365)
    elif tf.endswith("m"):
        mins = int(tf[:-1])
        periods_per_year = int(60 / mins * 24 * 365)
    elif tf.endswith("d"):
        periods_per_year = 365 // int(tf[:-1]) if tf[:-1].isdigit() else 365
    else:
        periods_per_year = 365 * 24

    model = None
    last_trained_at = None

    # For baseline buy-and-hold on test period
    try:
        bh_entry_price = float(df.iloc[test_start_idx]["open"])
        bh_exit_price = float(df.iloc[test_end_idx]["close"])
        bh_final = args.initial_capital * (bh_exit_price / bh_entry_price)
    except Exception:
        bh_final = None

    for idx in tqdm(range(test_start_idx, test_end_idx), desc="Walk-forward"):
        # train or reuse model
        if model is None or ((idx - (last_trained_at or train_end_idx)) >= args.retrain_every):
            # prepare history up to idx (exclusive of idx)
            hist_close = close.iloc[:idx]

            input_len = args.input_chunk_length
            output_len = args.forecast_steps
            min_required = input_len + output_len

            # decide whether we can train on returns
            hist_returns = hist_close.pct_change().dropna()
            if len(hist_returns) >= min_required and args.train_on_returns:
                predict_on_returns = True
                ts = TimeSeries.from_series(hist_returns)
                if args.debug:
                    print(f"Training on returns (len={len(ts)})")
            else:
                predict_on_returns = False
                ts = TimeSeries.from_series(hist_close)
                if args.debug and args.train_on_returns:
                    print(f"Not enough returns history ({len(hist_returns)}) for returns-train; training on prices")

            # initialize model with quantile likelihood (probabilistic)
            pl_trainer_kwargs = {"accelerator": "gpu" if args.device == "cuda" else "cpu", "devices": 1}

            model = NBEATSModel(
                input_chunk_length=input_len,
                output_chunk_length=output_len,
                generic_architecture=True,
                n_epochs=args.epochs,
                random_state=42,
                force_reset=True,
                likelihood=QuantileRegression(quantiles=[args.low_q, 0.5, args.high_q]),
                pl_trainer_kwargs=pl_trainer_kwargs,
            )

            # decide validation split robustly
            if len(ts) >= 2 * min_required:
                # choose val_points to be at least min_required
                desired_val = max(args.val_points, min_required)
                # ensure there's enough left for training
                val_points = min(desired_val, len(ts) - min_required)
                if val_points < min_required:
                    # fallback: keep val_points = min_required if possible
                    val_points = min_required
                    if len(ts) - val_points < min_required:
                        # not enough for train+val -- train without val
                        train_ts = ts
                        print("Warning: not enough history for train+val split; training without validation.")
                        model.fit(train_ts)
                        last_trained_at = idx
                        continue
                train_ts, val_ts = ts[:-val_points], ts[-val_points:]
                model.fit(train_ts, val_series=val_ts)
            else:
                # not enough length to have both train and val of required size -> train on all history without val
                print("Info: history shorter than 2*(input+output). Training on available history without validation.")
                train_ts = ts
                model.fit(train_ts)

            last_trained_at = idx

        # predict samples for forecast_steps
        prob_pred = model.predict(n=args.forecast_steps, num_samples=args.num_samples)

        # try to extract quantiles via new API
        arr = None
        median_pred = None
        low_pred = None
        high_pred = None
        pred0_timestamp = None
        try:
            q_low_ts = prob_pred.quantile_timeseries(args.low_q)
            q_mid_ts = prob_pred.quantile_timeseries(0.5)
            q_high_ts = prob_pred.quantile_timeseries(args.high_q)

            pred0_timestamp = q_mid_ts.time_index[0]
            median_pred = float(q_mid_ts.pd_series().iloc[0])
            low_pred = float(q_low_ts.pd_series().iloc[0])
            high_pred = float(q_high_ts.pd_series().iloc[0])

            # try to get samples too (for p_pos). Not all versions expose them; use helper
            try:
                arr = extract_samples_from_prob_pred(prob_pred)
            except Exception:
                arr = None

        except Exception:
            # Fallback for older Darts: compute quantiles from samples
            arr = extract_samples_from_prob_pred(prob_pred)
            arr = np.asarray(arr)

            # normalize shape to (samples, length)
            if arr.ndim == 3:
                samples, comps, length = arr.shape
                arr = arr[:, 0, :]
            elif arr.ndim == 2:
                pass
            else:
                raise RuntimeError(f"Unexpected shape from prob_pred.all_values(): {arr.shape}")

            q_low = np.quantile(arr, args.low_q, axis=0)
            q_mid = np.quantile(arr, 0.5, axis=0)
            q_high = np.quantile(arr, args.high_q, axis=0)

            median_pred = float(q_mid[0])
            low_pred = float(q_low[0])
            high_pred = float(q_high[0])
            pred0_timestamp = getattr(prob_pred, "time_index", [df.index[idx]])[0]

        # compute returns vs last observed close (decision uses last known close)
        last_price = float(close.iloc[idx - 1])

        # If we have raw samples (arr), convert them to returns for p_pos computation and robust quantiles
        p_pos = None
        if arr is not None:
            arr = np.asarray(arr)
            if arr.ndim == 3:
                arr = arr[:, 0, :]
            # samples for first step
            samples_first = arr[:, 0]
            if predict_on_returns:
                sample_returns = samples_first
            else:
                # convert price samples to returns
                sample_returns = (samples_first - last_price) / last_price

            # recompute median/quantiles from sample_returns if needed
            median_ret = float(np.median(sample_returns))
            low_ret = float(np.quantile(sample_returns, args.low_q))
            high_ret = float(np.quantile(sample_returns, args.high_q))

            p_pos = float((sample_returns > 0.0).mean())

            # keep median_pred in consistent form (return)
            median_pred = median_ret if predict_on_returns else (median_ret * last_price + last_price)

        else:
            # no raw samples available: fallback
            if predict_on_returns:
                median_ret = float(median_pred)
                low_ret = float(low_pred)
                high_ret = float(high_pred)
            else:
                median_ret = (float(median_pred) - last_price) / last_price
                low_ret = (float(low_pred) - last_price) / last_price
                high_ret = (float(high_pred) - last_price) / last_price

        # Decision: require both magnitude and probability (if available)
        signal = "HOLD"
        thr = args.threshold_pct
        # prefer probabilistic gating if samples available
        if p_pos is not None:
            if p_pos >= args.prob_threshold and median_ret >= thr:
                # strong vs normal depending on lower quantile positive
                signal = "STRONG_BUY" if low_ret > 0 else "BUY"
            elif p_pos <= (1.0 - args.prob_threshold) and median_ret <= -thr:
                signal = "STRONG_SELL" if high_ret < 0 else "SELL"
            else:
                signal = "HOLD"
        else:
            # fallback to quantile-only rule
            signal = decide_signal_from_quantiles(median_ret, low_ret, high_ret, thr)

        pos_frac = DEFAULT_POS_SIZING.get(signal, 0.0)
        allocation = abs(pos_frac) * capital

        # entry price: next bar OPEN (if available)
        entry_row = df.iloc[idx]
        entry_price = float(entry_row["open"]) if "open" in df.columns else float(entry_row["close"])
        units = allocation / entry_price if entry_price > 0 else 0.0
        side = "LONG" if pos_frac > 0 else ("SHORT" if pos_frac < 0 else "NONE")

        # exit after hold_bars at close
        exit_idx = idx + args.hold_bars
        exit_price = float(df.iloc[exit_idx]["close"])

        if side == "LONG":
            gross_pnl = (exit_price - entry_price) * units
        elif side == "SHORT":
            gross_pnl = (entry_price - exit_price) * units
        else:
            gross_pnl = 0.0

        fee = (abs(units) * entry_price) * args.taker_fee
        fee += (abs(units) * exit_price) * args.taker_fee
        slippage_cost = (abs(units) * entry_price) * args.slippage + (abs(units) * exit_price) * args.slippage
        net_pnl = gross_pnl - fee - slippage_cost

        if side != "NONE" and abs(net_pnl) > 1e-12:
            capital += net_pnl

        equity_ts.append(capital)
        equity_time.append(df.index[exit_idx])

        trades.append({
            "decision_time": df.index[idx - 1],
            "signal_time": pred0_timestamp,
            "signal": signal,
            "side": side,
            "entry_time": df.index[idx],
            "entry_price": entry_price,
            "exit_time": df.index[exit_idx],
            "exit_price": exit_price,
            "units": units,
            "allocation_usd": allocation,
            "gross_pnl": gross_pnl,
            "fee_usd": fee,
            "slippage_usd": slippage_cost,
            "net_pnl": net_pnl,
            "capital_after": capital,
            "median_pred": median_pred,
            "low_pred": low_pred,
            "high_pred": high_pred,
            "median_ret": median_ret,
            "low_ret": low_ret,
            "high_ret": high_ret,
            "p_pos": p_pos,
            "predict_on_returns": predict_on_returns,
        })

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame({"timestamp": equity_time, "equity": equity_ts}).set_index("timestamp")

    # attach baseline
    return trades_df, equity_df, capital, periods_per_year, bh_final


# ---------------------
# Main CLI
# ---------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["backtest", "live_sim"], default="backtest")
    p.add_argument("--symbol", default="BTC/USDT")
    p.add_argument("--timeframe", default="1h")
    p.add_argument("--max_bars", type=int, default=2000)
    p.add_argument("--train_frac", type=float, default=0.6)
    p.add_argument("--val_points", type=int, default=168)
    p.add_argument("--input_chunk_length", type=int, default=168)
    p.add_argument("--forecast_steps", type=int, default=1)
    p.add_argument("--hold_bars", type=int, default=1)
    p.add_argument("--epochs", type=int, default=6,
                   help="N-BEATS training epochs per retrain (small for fast iteration)")
    p.add_argument("--retrain_every", type=int, default=24,
                   help="retrain model every N test steps (set large to reduce retrain frequency)")
    p.add_argument("--num_samples", type=int, default=100)
    p.add_argument("--low_q", type=float, default=0.10)
    p.add_argument("--high_q", type=float, default=0.90)
    p.add_argument("--threshold_pct", type=float, default=0.002, help="decision threshold (e.g. 0.002 = 0.2%)")
    p.add_argument("--prob_threshold", type=float, default=0.60, help="required probability mass for directional trade")
    p.add_argument("--train_on_returns", action="store_true", help="force training on returns when possible (recommended)")
    p.add_argument("--initial_capital", type=float, default=1000.0)
    p.add_argument("--taker_fee", type=float, default=0.001, help="per-trade fee fraction (default 0.1%)")
    p.add_argument("--slippage", type=float, default=0.0005, help="per-trade slippage fraction (default 0.05%)")
    p.add_argument("--torch_threads", type=int, default=0, help="set PyTorch intra-op threads (0 = don't set)")
    p.add_argument("--output_dir", default="./paper_out")
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # detect device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device
    if args.torch_threads and args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)

    print(f"PyTorch device: {device}, torch threads: {torch.get_num_threads()}")
    if device == "cuda":
        print(f"CUDA available. GPU name: {torch.cuda.get_device_name(0)}")

    exchange = ccxt.binance({"enableRateLimit": True})
    print("Fetching historical OHLCV...")
    df = fetch_ohlcv_full(exchange, symbol=args.symbol, timeframe=args.timeframe, max_bars=args.max_bars)
    print(f"Fetched {len(df)} bars from {df.index[0]} to {df.index[-1]}")

    for col in ["open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            raise RuntimeError(f"Missing column {col} in fetched data")

    trades_df, equity_df, final_capital, periods_per_year, bh_final = walk_forward_backtest(df, args)

    returns = equity_df["equity"].pct_change().fillna(0).values
    ann_sharpe = annualized_sharpe(returns, periods_per_year)
    mdd = max_drawdown(equity_df["equity"].values) if not equity_df.empty else 0.0
    total_return = (final_capital / args.initial_capital - 1.0) * 100.0

    print("\n--- BACKTEST SUMMARY ---")
    print(f"Initial capital: ${args.initial_capital:.2f}")
    print(f"Final capital:   ${final_capital:.2f}")
    print(f"Total return:    {total_return:.2f}%")
    print(f"Ann. Sharpe (est): {ann_sharpe:.3f}")
    print(f"Max Drawdown:    {mdd*100:.2f}%")
    print(f"Number of trades: {len(trades_df)}")
    if len(trades_df):
        wins = (trades_df["net_pnl"] > 0).sum()
        print(f"Winning trades: {wins} / {len(trades_df)} ({wins/len(trades_df)*100:.1f}%)")
        print(f"Avg net PnL per trade: ${trades_df['net_pnl'].mean():.2f}")

    if bh_final is not None:
        bh_pct = (bh_final / args.initial_capital - 1.0) * 100.0
        print(f"Buy-and-hold final capital (test period): ${bh_final:.2f} ({bh_pct:.2f}%)")

    trades_csv = out_dir / "trades_log.csv"
    equity_csv = out_dir / "equity_curve.csv"
    trades_df.to_csv(trades_csv, index=False)
    equity_df.to_csv(equity_csv)

    plt.figure(figsize=(10, 5))
    if not equity_df.empty:
        plt.plot(equity_df.index, equity_df["equity"], label="equity")
    plt.title("Paper-trade equity curve")
    plt.xlabel("time")
    plt.ylabel("USD")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plot_file = out_dir / "equity_curve.png"
    plt.savefig(plot_file)
    plt.show()

    print("\nSaved files to:", out_dir)
    print("Trades CSV:", trades_csv)
    print("Equity CSV:", equity_csv)


if __name__ == "__main__":
    main()
