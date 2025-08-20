#!/usr/bin/env python3
"""
paper_nbeats_papertrade_gpu_patched_fixed_v7.py

v7: Adds automatic hyper-optimization using Optuna (optional), --run_best flag,
better executed-trades accounting, safer defaults for min_allocation during search,
and a convenient `--auto_tune` mode that runs a short Optuna search and then
re-runs the backtest with the best found configuration.

Important notes:
 - This does NOT guarantee live profitability. It only automates searching the
   parameter space (with walk-forward backtests) and returns the best config
   found according to the chosen objective (default: ann_sharpe).
 - For speed during optimization, the script will reduce n_epochs and num_samples
   unless you explicitly set higher values. Always validate the returned best
   configuration with a full, slow backtest (higher epochs/num_samples).

Usage examples:
 - Quick auto-tune (fast trial runs):
   python paper_nbeats_papertrade_gpu_patched_fixed_v7.py --use_optuna --optuna_trials 20 --max_bars 2000 --timeframe 4h --output_dir ./optuna_out

 - Full evaluation with the best config saved by optuna:
   python paper_nbeats_papertrade_gpu_patched_fixed_v7.py --run_best --output_dir ./best_run_out --max_bars 8000 --timeframe 4h

Requirements (optional): optuna, PyWavelets
  pip install optuna PyWavelets

"""

from darts.utils.likelihood_models import QuantileRegression
from darts.dataprocessing.transformers import Scaler
from darts import TimeSeries
from darts.models import NBEATSModel

import argparse
import time
from pathlib import Path
import math
import warnings
import requests
import json
import sys

import ccxt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import itertools
import random

warnings.filterwarnings("ignore")

try:
    import pywt
    HAS_PYWT = True
except Exception:
    HAS_PYWT = False

try:
    import optuna
    HAS_OPTUNA = True
except Exception:
    HAS_OPTUNA = False


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


DEFAULT_POS_SIZING = {
    "STRONG_BUY": 0.06,
    "BUY": 0.03,
    "HOLD": 0.0,
    "SELL": -0.03,
    "STRONG_SELL": -0.06,
}


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


def extract_samples_from_prob_pred(prob_pred):
    getter = getattr(prob_pred, "all_values", None)
    if callable(getter):
        arr = getter()
        return np.asarray(arr)

    getter = getattr(prob_pred, "samples", None)
    if callable(getter):
        arr = getter()
        return np.asarray(arr)

    arr = getattr(prob_pred, "_all_values", None)
    if arr is not None:
        return np.asarray(arr)

    raise RuntimeError("Cannot extract samples from probabilistic prediction (no all_values/samples API)")


# ---------------------
# Wavelet helpers
# ---------------------

def add_wavelet_features(df, col='close', level=3, wavelet='db4'):
    if not HAS_PYWT:
        return df
    arr = df[col].values
    try:
        coeffs = pywt.wavedec(arr, wavelet, level=level)
    except Exception:
        return df
    for i, c in enumerate(coeffs):
        c_up = np.interp(np.arange(len(arr)), np.linspace(0, len(arr)-1, num=len(c)), c)
        df[f'wav_{i}'] = c_up
    return df


# ---------------------
# Backtest core
# ---------------------

def walk_forward_backtest(df, args, return_all_trades=True):
    close = df["close"]
    n = len(df)
    train_size = int(args.train_frac * n)
    if train_size < 100:
        raise ValueError("Train fraction too small or not enough bars; increase max_bars or train_frac")

    train_end_idx = train_size
    test_start_idx = train_end_idx
    test_end_idx = n - args.hold_bars

    capital = float(args.initial_capital)
    equity_ts = []
    equity_time = []
    trades = []

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

    try:
        bh_entry_price = float(df.iloc[test_start_idx]["open"])
        bh_exit_price = float(df.iloc[test_end_idx]["close"])
        bh_final = args.initial_capital * (bh_exit_price / bh_entry_price)
    except Exception:
        bh_final = None

    last_trade_idx = -9999
    last_exit_idx = -9999

    df['tr'] = (df['high'] - df['low']).abs()
    df['atr_14'] = df['tr'].rolling(14, min_periods=1).mean()

    atr_window = min(500, max(50, int(len(df) * 0.2)))
    df['atr_rank'] = df['atr_14'].rolling(atr_window, min_periods=1).apply(lambda x: (np.searchsorted(np.sort(x), x[-1]) / len(x) * 100), raw=False)

    if args.use_wavelet:
        df = add_wavelet_features(df, col='close', level=args.wavelet_levels, wavelet=args.wavelet)

    cov_features = []
    if args.use_indicators:
        cov_features = ["ema_50", "ema_200", "rsi_14", "atr_14", "volume", "fng", "dayofweek_sin", "dayofweek_cos", "month_sin", "month_cos"]

    if args.use_wavelet and HAS_PYWT:
        wav_cols = [c for c in df.columns if c.startswith('wav_')]
        cov_features += wav_cols

    # Track executed trades separately for clear metrics
    executed_trades = []

    for idx in tqdm(range(test_start_idx, test_end_idx), desc="Walk-forward"):
        if model is None or ((idx - (last_trained_at or train_end_idx)) >= args.retrain_every):
            hist_df = df.iloc[:idx].copy()

            input_len = args.input_chunk_length
            output_len = args.forecast_steps
            min_required = input_len + output_len

            hist_close = hist_df["close"]
            hist_returns = hist_close.pct_change().dropna()
            if args.train_on_returns and len(hist_returns) >= min_required:
                predict_on_returns = True
                target_ts = TimeSeries.from_series(hist_returns)
            else:
                predict_on_returns = False
                target_ts = TimeSeries.from_series(hist_close)

            cov_ts = None
            cov_train = cov_val = None
            if cov_features:
                cov_df = hist_df[cov_features].copy()
                cov_ts = TimeSeries.from_dataframe(cov_df)

            # reduce training expense if running inside optuna search
            trainer_kwargs = {"accelerator": "gpu" if args.device == "cuda" else "cpu", "devices": 1}

            model = NBEATSModel(
                input_chunk_length=input_len,
                output_chunk_length=output_len,
                generic_architecture=True,
                num_blocks=args.num_blocks,
                num_layers=args.num_layers,
                layer_widths=args.layer_widths,
                n_epochs=args.epochs,
                random_state=42,
                force_reset=True,
                likelihood=QuantileRegression(quantiles=[args.low_q, 0.5, args.high_q]),
                pl_trainer_kwargs=trainer_kwargs,
                batch_size=args.batch_size,
            )

            scaler = None
            if not predict_on_returns:
                scaler = Scaler()

            if len(target_ts) >= 2 * min_required:
                desired_val = max(args.val_points, min_required)
                val_points = min(desired_val, len(target_ts) - min_required)
                if val_points < min_required:
                    train_ts = target_ts
                    if scaler:
                        train_ts = scaler.fit_transform(train_ts)
                    model.fit(train_ts, past_covariates=cov_ts)
                    last_trained_at = idx
                    continue
                train_ts, val_ts = target_ts[:-val_points], target_ts[-val_points:]
                cov_train = cov_ts[:-val_points] if cov_ts else None
                cov_val = cov_ts[-val_points:] if cov_ts else None
                if scaler:
                    train_ts = scaler.fit_transform(train_ts)
                    val_ts = scaler.transform(val_ts)
                model.fit(train_ts, val_series=val_ts,
                          past_covariates=cov_train if cov_ts is not None else None,
                          val_past_covariates=cov_val if cov_ts is not None else None)
            else:
                train_ts = target_ts
                if scaler:
                    train_ts = scaler.fit_transform(train_ts)
                model.fit(train_ts, past_covariates=cov_ts)

            last_trained_at = idx

        cov_ts_pred = None
        if cov_features:
            cov_df_here = df[cov_features].iloc[:idx].copy()
            cov_ts_pred = TimeSeries.from_dataframe(cov_df_here)

        prob_pred = model.predict(n=args.forecast_steps, num_samples=args.num_samples, past_covariates=cov_ts_pred)

        if not predict_on_returns and scaler:
            prob_pred = scaler.inverse_transform(prob_pred)

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

            try:
                arr = extract_samples_from_prob_pred(prob_pred)
            except Exception:
                arr = None

        except Exception:
            arr = extract_samples_from_prob_pred(prob_pred)
            arr = np.asarray(arr)

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

        last_price = float(close.iloc[idx - 1])

        p_pos = None
        if arr is not None:
            arr = np.asarray(arr)
            if arr.ndim == 3:
                arr = arr[:, 0, :]
            samples_first = arr[:, 0]
            if predict_on_returns:
                sample_returns = samples_first
            else:
                sample_returns = (samples_first - last_price) / last_price

            thr = args.threshold_pct
            p_pos = float((sample_returns > thr).mean())

            median_ret = float(np.median(sample_returns))
            low_ret = float(np.quantile(sample_returns, args.low_q))
            high_ret = float(np.quantile(sample_returns, args.high_q))

            median_pred = median_ret if predict_on_returns else (median_ret * last_price + last_price)

        else:
            if predict_on_returns:
                median_ret = float(median_pred)
                low_ret = float(low_pred)
                high_ret = float(high_pred)
            else:
                median_ret = (float(median_pred) - last_price) / last_price
                low_ret = (float(low_pred) - last_price) / last_price
                high_ret = (float(high_pred) - last_price) / last_price

        signal = "HOLD"
        thr = args.threshold_pct
        if p_pos is not None:
            if p_pos >= args.prob_threshold and median_ret >= thr:
                signal = "STRONG_BUY" if low_ret > 0 else "BUY"
            elif p_pos <= (1.0 - args.prob_threshold) and median_ret <= -thr:
                signal = "STRONG_SELL" if high_ret < 0 else "SELL"
            else:
                signal = "HOLD"
        else:
            signal = decide_signal_from_quantiles(median_ret, low_ret, high_ret, thr)

        if args.require_trend and "ema_50" in df.columns and "ema_200" in df.columns:
            in_uptrend = df["ema_50"].iloc[idx - 1] > df["ema_200"].iloc[idx - 1]
            if signal in ("BUY", "STRONG_BUY") and not in_uptrend:
                signal = "HOLD"
            if signal in ("SELL", "STRONG_SELL") and in_uptrend:
                signal = "HOLD"

        if "rsi_14" in df.columns:
            rsi = df["rsi_14"].iloc[idx - 1]
            if signal in ("BUY", "STRONG_BUY") and rsi > 70:
                signal = "HOLD"
            if signal in ("SELL", "STRONG_SELL") and rsi < 30:
                signal = "HOLD"

        if args.atr_max_percentile is not None:
            atr_pct = df['atr_rank'].iloc[idx - 1] if 'atr_rank' in df.columns else 0
            if atr_pct >= args.atr_max_percentile:
                signal = "HOLD"

        pos_frac = DEFAULT_POS_SIZING.get(signal, 0.0)

        if args.scale_by_confidence and p_pos is not None and pos_frac != 0.0:
            if pos_frac > 0:
                scale = (p_pos - args.prob_threshold) / (1.0 - args.prob_threshold)
            else:
                scale = ((1.0 - args.prob_threshold) - p_pos) / (1.0 - args.prob_threshold)
            scale = max(-1.0, min(1.0, scale))
            multiplier = 1.0 + 0.5 * scale
            pos_frac = pos_frac * multiplier

        if idx < last_exit_idx:
            signal = "HOLD"
            pos_frac = 0.0

        allocation = abs(pos_frac) * capital
        entry_row = df.iloc[idx]
        entry_price = float(entry_row["open"]) if "open" in df.columns else float(entry_row["close"])

        units = 0.0
        side = "NONE"
        if pos_frac != 0.0:
            if args.use_vol_sizing and 'atr_14' in df.columns and df['atr_14'].iloc[idx - 1] > 0:
                atr_here = df['atr_14'].iloc[idx - 1]
                risk_per_unit = args.atr_multiplier_stop * atr_here
                dollars_risk = capital * args.risk_per_trade
                if risk_per_unit <= 0:
                    units = 0.0
                else:
                    units = dollars_risk / risk_per_unit
                    max_affordable_units = (capital * abs(pos_frac)) / entry_price
                    units = min(units, max_affordable_units)
            else:
                allocation = abs(pos_frac) * capital
                units = allocation / entry_price if entry_price > 0 else 0.0

            side = "LONG" if pos_frac > 0 else ("SHORT" if pos_frac < 0 else "NONE")

        allocation_usd = units * entry_price
        if allocation_usd < args.min_allocation or (idx - last_trade_idx) < args.min_bars_between_trades:
            side = "NONE"
            units = 0.0
            allocation_usd = 0.0

        exit_idx = idx + args.hold_bars
        exit_price = float(df.iloc[exit_idx]["close"]) if exit_idx < len(df) else float(df.iloc[-1]["close"])
        exit_reason = "time_exit"
        gross_pnl = 0.0

        if side == "LONG" and units > 0:
            entry_price = float(df.iloc[idx]["open"]) if "open" in df.columns else float(df.iloc[idx]["close"])
            atr_init = df['atr_14'].iloc[idx - 1] if 'atr_14' in df.columns else None
            sl_price = entry_price - args.atr_multiplier_stop * atr_init if atr_init is not None else (entry_price * (1.0 - args.stop_loss) if args.stop_loss else None)
            tp_price = entry_price * (1.0 + args.take_profit) if args.take_profit and args.take_profit > 0 else None

            highest_price = entry_price
            trailing_sl = sl_price

            for j in range(idx, min(exit_idx + 1, len(df)-1)):
                row = df.iloc[j]
                low_j = float(row["low"]) if "low" in df.columns else float(row["close"])
                high_j = float(row["high"]) if "high" in df.columns else float(row["close"])

                if high_j > highest_price:
                    highest_price = high_j
                    atr_j = row['atr_14'] if 'atr_14' in row.index else atr_init
                    trailing_sl = highest_price - args.atr_multiplier_stop * (atr_j if not np.isnan(atr_j) else atr_init)

                if trailing_sl is not None and low_j <= trailing_sl:
                    exit_price = trailing_sl
                    exit_idx = j
                    exit_reason = "trailing_sl"
                    break

                if sl_price is not None and low_j <= sl_price:
                    exit_price = sl_price
                    exit_idx = j
                    exit_reason = "sl"
                    break

                if tp_price is not None and high_j >= tp_price:
                    exit_price = tp_price
                    exit_idx = j
                    exit_reason = "tp"
                    break

            gross_pnl = (exit_price - entry_price) * units

        elif side == "SHORT" and units > 0:
            entry_price = float(df.iloc[idx]["open"]) if "open" in df.columns else float(df.iloc[idx]["close"])
            atr_init = df['atr_14'].iloc[idx - 1] if 'atr_14' in df.columns else None
            sl_price = entry_price + args.atr_multiplier_stop * atr_init if atr_init is not None else (entry_price * (1.0 + args.stop_loss) if args.stop_loss else None)
            tp_price = entry_price * (1.0 - args.take_profit) if args.take_profit and args.take_profit > 0 else None

            lowest_price = entry_price
            trailing_sl = sl_price

            for j in range(idx, min(exit_idx + 1, len(df)-1)):
                row = df.iloc[j]
                low_j = float(row["low"]) if "low" in df.columns else float(row["close"])
                high_j = float(row["high"]) if "high" in df.columns else float(row["close"])

                if low_j < lowest_price:
                    lowest_price = low_j
                    atr_j = row['atr_14'] if 'atr_14' in row.index else atr_init
                    trailing_sl = lowest_price + args.atr_multiplier_stop * (atr_j if not np.isnan(atr_j) else atr_init)

                if trailing_sl is not None and high_j >= trailing_sl:
                    exit_price = trailing_sl
                    exit_idx = j
                    exit_reason = "trailing_sl"
                    break

                if sl_price is not None and high_j >= sl_price:
                    exit_price = sl_price
                    exit_idx = j
                    exit_reason = "sl"
                    break

                if tp_price is not None and low_j <= tp_price:
                    exit_price = tp_price
                    exit_idx = j
                    exit_reason = "tp"
                    break

            gross_pnl = (entry_price - exit_price) * units
        else:
            gross_pnl = 0.0
            exit_reason = "none"

        fee = (abs(units) * entry_price) * args.taker_fee if units else 0.0
        fee += (abs(units) * exit_price) * args.taker_fee if units else 0.0
        slippage_cost = (abs(units) * entry_price) * args.slippage + (abs(units) * exit_price) * args.slippage if units else 0.0
        net_pnl = gross_pnl - fee - slippage_cost

        if args.debug and side != "NONE":
            print(f"TRADE idx={idx} signal={signal} p_pos={p_pos:.3f} median_ret={median_ret:.4f} alloc=${allocation_usd:.2f} units={units:.6f} exit_reason={exit_reason}")

        if side != "NONE" and abs(net_pnl) > 1e-12:
            capital += net_pnl
            last_trade_idx = idx
            last_exit_idx = exit_idx

        equity_ts.append(capital)
        equity_time.append(df.index[exit_idx] if exit_idx < len(df) else df.index[-1])

        trade_record = {
            "decision_time": df.index[idx - 1],
            "signal_time": pred0_timestamp,
            "signal": signal,
            "side": side,
            "entry_time": df.index[idx] if side != "NONE" else pd.NaT,
            "entry_price": entry_price if side != "NONE" else np.nan,
            "exit_time": df.index[exit_idx] if side != "NONE" else pd.NaT,
            "exit_price": exit_price if side != "NONE" else np.nan,
            "units": units,
            "allocation_usd": allocation_usd,
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
            "exit_reason": exit_reason,
        }

        trades.append(trade_record)
        if trade_record['side'] != 'NONE' and trade_record['units'] > 0:
            executed_trades.append(trade_record)

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame({"timestamp": equity_time, "equity": equity_ts}).set_index("timestamp")
    exec_df = pd.DataFrame(executed_trades)

    if return_all_trades:
        return trades_df, equity_df, capital, periods_per_year, bh_final
    else:
        return exec_df, equity_df, capital, periods_per_year, bh_final


# ---------------------
# Grid / optuna utils
# ---------------------

def evaluate_result(trades_df, equity_df, final_capital, args, periods_per_year):
    result = {}
    result['final_capital'] = float(final_capital)
    result['total_return_pct'] = float((final_capital / args.initial_capital - 1.0) * 100.0)
    returns = equity_df['equity'].pct_change().fillna(0).values
    result['ann_sharpe'] = float(annualized_sharpe(returns, periods_per_year))
    result['max_drawdown_pct'] = float(max_drawdown(equity_df['equity'].values) * 100.0) if not equity_df.empty else np.nan
    # focus on executed trades
    exec_df = trades_df[trades_df['side'] != 'NONE'] if not trades_df.empty else trades_df
    result['num_executed'] = int(len(exec_df))
    if len(exec_df):
        wins = (exec_df['net_pnl'] > 0).sum()
        result['win_rate'] = float(wins / len(exec_df))
        result['avg_pnl'] = float(exec_df['net_pnl'].mean())
        gross_wins = exec_df.loc[exec_df['net_pnl'] > 0, 'net_pnl'].sum()
        gross_losses = -exec_df.loc[exec_df['net_pnl'] < 0, 'net_pnl'].sum()
        result['profit_factor'] = float(gross_wins / gross_losses) if gross_losses > 0 else np.nan
        try:
            boot_means = []
            arr = exec_df['net_pnl'].values
            for _ in range(300):
                sample = np.random.choice(arr, size=len(arr), replace=True)
                boot_means.append(np.mean(sample))
            result['avg_pnl_ci_l'] = float(np.percentile(boot_means, 2.5))
            result['avg_pnl_ci_u'] = float(np.percentile(boot_means, 97.5))
        except Exception:
            result['avg_pnl_ci_l'] = np.nan
            result['avg_pnl_ci_u'] = np.nan
    else:
        result.update({'win_rate': np.nan, 'avg_pnl': np.nan, 'profit_factor': np.nan, 'avg_pnl_ci_l': np.nan, 'avg_pnl_ci_u': np.nan})

    return result


def build_param_grid():
    grid = {
        'prob_threshold': [0.85, 0.9, 0.95],
        'threshold_pct': [0.002, 0.005, 0.01],
        'atr_multiplier_stop': [1.5, 2.0],
        'risk_per_trade': [0.005, 0.01],
        'input_chunk_length': [48, 72],
        'use_wavelet': [False, True],
        'use_vol_sizing': [False, True],
        'num_blocks': [2, 3],
        'num_layers': [3, 4],
        'layer_widths': [64, 128],
    }
    keys = list(grid.keys())
    all_vals = list(itertools.product(*(grid[k] for k in keys)))
    combos = [dict(zip(keys, vals)) for vals in all_vals]
    return combos


def run_grid_search(df, base_args, out_dir, sample_size=50, seed=42, objective='ann_sharpe', verbose=True):
    combos = build_param_grid()
    total = len(combos)
    random.seed(seed)
    random.shuffle(combos)
    if sample_size is not None and sample_size < total:
        combos = combos[:sample_size]
    results = []

    for i, combo in enumerate(combos):
        print(f"Grid {i+1}/{len(combos)}: {combo}")
        a = argparse.Namespace(**vars(base_args))
        for k, v in combo.items():
            setattr(a, k, v)
        # reduce min_allocation during search so small allocs run
        a.min_allocation = min(a.min_allocation, 5.0)
        run_label = f"run_{i+1}"
        a.output_dir = str(Path(out_dir) / run_label)
        Path(a.output_dir).mkdir(parents=True, exist_ok=True)

        try:
            trades_df, equity_df, final_capital, periods_per_year, bh_final = walk_forward_backtest(df.copy(), a)
            metrics = evaluate_result(trades_df, equity_df, final_capital, a, periods_per_year)
            metrics.update(combo)
            metrics['grid_idx'] = i + 1
            results.append(metrics)
            df_results = pd.DataFrame(results)
            df_results.to_csv(Path(out_dir) / 'grid_results_partial.csv', index=False)
        except Exception as e:
            print(f"Grid run failed: {e}")
            results.append({'grid_idx': i+1, 'error': str(e), **combo})

    df_results = pd.DataFrame(results)
    if 'error' in df_results.columns:
        df_results.to_csv(Path(out_dir) / 'grid_results_with_errors.csv', index=False)
    df_results.to_csv(Path(out_dir) / 'grid_results.csv', index=False)

    if objective not in df_results.columns:
        objective = 'ann_sharpe'
    df_sorted = df_results.sort_values(by=objective, ascending=False)
    best = df_sorted.iloc[0].to_dict()
    best_config = {k: best[k] for k in best.keys() if k in combo.keys()}
    with open(Path(out_dir) / 'best_config.json', 'w') as fh:
        json.dump(best_config, fh, indent=2)

    if verbose:
        print('--- GRID SEARCH COMPLETE ---')
        print('Results saved to', out_dir)
        print('Top 5 configs:')
        print(df_sorted.head(5).to_string(index=False))

    return df_results, best_config


def run_optuna_search(df, base_args, out_dir, n_trials=30, objective='ann_sharpe', seed=42, timeout=None):
    if not HAS_OPTUNA:
        raise RuntimeError('optuna not installed. Install with: pip install optuna')

    def suggest_params(trial):
        return {
            'prob_threshold': trial.suggest_categorical('prob_threshold', [0.85, 0.9, 0.95]),
            'threshold_pct': trial.suggest_categorical('threshold_pct', [0.002, 0.005, 0.01]),
            'atr_multiplier_stop': trial.suggest_categorical('atr_multiplier_stop', [1.5, 2.0, 2.5]),
            'risk_per_trade': trial.suggest_categorical('risk_per_trade', [0.005, 0.01]),
            'input_chunk_length': trial.suggest_categorical('input_chunk_length', [48, 72]),
            'use_wavelet': trial.suggest_categorical('use_wavelet', [False, True]),
            'use_vol_sizing': trial.suggest_categorical('use_vol_sizing', [False, True]),
            'num_blocks': trial.suggest_categorical('num_blocks', [2, 3]),
            'num_layers': trial.suggest_categorical('num_layers', [3, 4]),
            'layer_widths': trial.suggest_categorical('layer_widths', [64, 128]),
        }

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=seed))

    def objective_fn(trial):
        combo = suggest_params(trial)
        a = argparse.Namespace(**vars(base_args))
        for k, v in combo.items():
            setattr(a, k, v)
        # make tuning fast by reducing epochs and num_samples unless user specified high values
        if not getattr(base_args, '_user_set_epochs', False):
            a.epochs = max(3, min(10, a.epochs))
        if not getattr(base_args, '_user_set_num_samples', False):
            a.num_samples = max(50, min(200, a.num_samples))
        a.min_allocation = min(a.min_allocation, 5.0)

        try:
            trades_df, equity_df, final_capital, periods_per_year, bh_final = walk_forward_backtest(df.copy(), a)
            metrics = evaluate_result(trades_df, equity_df, final_capital, a, periods_per_year)
            score = metrics.get(objective, metrics.get('ann_sharpe', 0.0))
        except Exception as e:
            print('Optuna trial failed:', e)
            score = -9999.0
        trial.set_user_attr('metrics', metrics if 'metrics' in locals() else {})
        return score

    study.optimize(objective_fn, n_trials=n_trials, timeout=timeout)

    best = study.best_trial
    best_config = best.params
    # Save best config and some trial stats
    with open(Path(out_dir) / 'optuna_best_config.json', 'w') as fh:
        json.dump({'best_params': best_config, 'best_value': best.value}, fh, indent=2)

    # Dump study dataframe (small) for inspection
    df_trials = study.trials_dataframe().sort_values(by='value', ascending=False)
    df_trials.to_csv(Path(out_dir) / 'optuna_trials.csv', index=False)

    return study, best_config


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
    p.add_argument("--input_chunk_length", type=int, default=72)
    p.add_argument("--forecast_steps", type=int, default=1)
    p.add_argument("--hold_bars", type=int, default=1)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--retrain_every", type=int, default=24)
    p.add_argument("--num_samples", type=int, default=500)
    p.add_argument("--low_q", type=float, default=0.10)
    p.add_argument("--high_q", type=float, default=0.90)
    p.add_argument("--threshold_pct", type=float, default=0.01)
    p.add_argument("--prob_threshold", type=float, default=0.9)
    p.add_argument("--train_on_returns", action="store_true")
    p.add_argument("--initial_capital", type=float, default=1000.0)
    p.add_argument("--taker_fee", type=float, default=0.001)
    p.add_argument("--slippage", type=float, default=0.0005)
    p.add_argument("--torch_threads", type=int, default=0)
    p.add_argument("--min_allocation", type=float, default=50.0)
    p.add_argument("--min_bars_between_trades", type=int, default=12)
    p.add_argument("--stop_loss", type=float, default=0.03)
    p.add_argument("--take_profit", type=float, default=0.06)
    p.add_argument("--scale_by_confidence", action="store_true")
    p.add_argument("--use_indicators", action="store_true", help="enable indicators and covariates (default False)")
    p.add_argument("--require_trend", action="store_true")
    p.add_argument("--output_dir", default="./paper_out")
    p.add_argument("--debug", action="store_true")

    # new args
    p.add_argument("--risk_per_trade", type=float, default=0.01, help="fraction of equity risked per trade when using vol sizing")
    p.add_argument("--use_vol_sizing", action="store_true", help="use ATR-based volatility sizing instead of fixed fraction")
    p.add_argument("--atr_multiplier_stop", type=float, default=2.0, help="ATR multiplier for stop/trailing calculations")
    p.add_argument("--atr_max_percentile", type=float, default=90.0, help="skip entries when ATR percentile >= this (helps avoid extreme volatility)")
    p.add_argument("--use_wavelet", dest='use_wavelet', action='store_true', help="add wavelet multi-resolution covariates (requires pywt)")
    p.add_argument("--wavelet_levels", type=int, default=3)
    p.add_argument("--wavelet", type=str, default='db4')

    # model hyperparameters
    p.add_argument("--num_blocks", type=int, default=3)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--layer_widths", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=64)

    # grid/optuna options
    p.add_argument("--run_grid", action='store_true', help="run automated grid/random search")
    p.add_argument("--grid_sample", type=int, default=30, help="number of combinations to sample from the full grid (recommended)")
    p.add_argument("--grid_seed", type=int, default=42)
    p.add_argument("--grid_objective", type=str, default='ann_sharpe', help="metric to rank grid results (ann_sharpe, final_capital, total_return_pct)")

    p.add_argument("--use_optuna", action='store_true', help="use Optuna for smarter hyperparameter search (fast trials)")
    p.add_argument("--optuna_trials", type=int, default=30)
    p.add_argument("--optuna_timeout", type=int, default=None)
    p.add_argument("--optuna_seed", type=int, default=42)

    p.add_argument("--run_best", action='store_true', help="load best_config.json from output_dir and run final evaluation")

    args = p.parse_args()

    # detect if user explicitly set epochs or num_samples (to avoid Optuna changing them implicitly)
    args._user_set_epochs = '--epochs' in sys.argv
    args._user_set_num_samples = '--num_samples' in sys.argv

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device
    if args.torch_threads and args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)

    print(f"PyTorch device: {device}, torch threads: {torch.get_num_threads()}")
    if device == "cuda":
        try:
            print(f"CUDA available. GPU name: {torch.cuda.get_device_name(0)}")
        except Exception:
            pass

    exchange = ccxt.binance({"enableRateLimit": True})
    print("Fetching historical OHLCV...")
    df = fetch_ohlcv_full(exchange, symbol=args.symbol, timeframe=args.timeframe, max_bars=args.max_bars)
    # save the OHLCV used so calibrator can recompute future returns for arbitrary holds
    try:
        df.to_csv(Path(args.output_dir) / "ohlcv_used.csv")
    except Exception:
        pass

    print(f"Fetched {len(df)} bars from {df.index[0]} to {df.index[-1]}")

    for col in ["open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            raise RuntimeError(f"Missing column {col} in fetched data")

    print("Fetching historical Fear and Greed Index...")
    try:
        response = requests.get("https://api.alternative.me/fng/?limit=0", timeout=10)
        if response.status_code == 200:
            fng_data = response.json().get('data', [])
            fng_df = pd.DataFrame(fng_data)
            if not fng_df.empty:
                fng_df['timestamp'] = pd.to_datetime(fng_df['timestamp'], unit='s')
                fng_df = fng_df.set_index('timestamp')
                fng_df['fng'] = pd.to_numeric(fng_df['value'], errors='coerce') / 100.0
                fng_df = fng_df.resample('D').ffill()
                df = df.join(fng_df['fng'], how='left').fillna(method='ffill')
            else:
                df['fng'] = np.nan
        else:
            print("Failed to fetch FNG; proceeding without.")
            df['fng'] = np.nan
    except Exception as e:
        print("FNG fetch failed:", e)
        df['fng'] = np.nan

    df['dayofweek_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
    df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)

    if args.use_indicators:
        df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()
        df["ema_200"] = df["close"].ewm(span=200, adjust=False).mean()
        delta = df["close"].diff()
        up = delta.clip(lower=0).rolling(14).mean()
        down = -delta.clip(upper=0).rolling(14).mean()
        df["rsi_14"] = 100 - 100 / (1 + (up / down).replace([np.inf, -np.inf], np.nan))

    # run optuna/grid if requested
    if args.use_optuna:
        print('Starting Optuna search (this will run many fast trials).')
        study, best_config = run_optuna_search(df.copy(), args, out_dir, n_trials=args.optuna_trials, objective=args.grid_objective, seed=args.optuna_seed, timeout=args.optuna_timeout)
        print('Optuna best params saved to', Path(out_dir) / 'optuna_best_config.json')
        # automatically run final evaluation with the best params
        print('Re-running final evaluation with best optuna params (this will use your normal epochs & num_samples)')
        # merge best params into args and run full backtest
        for k, v in study.best_trial.params.items():
            setattr(args, k, v)
        # restore user epochs/num_samples if they were set
        if not args._user_set_epochs:
            args.epochs = 50
        if not args._user_set_num_samples:
            args.num_samples = 500
        trades_df, equity_df, final_capital, periods_per_year, bh_final = walk_forward_backtest(df, args)
        metrics = evaluate_result(trades_df, equity_df, final_capital, args, periods_per_year)
        print('Final evaluation metrics after Optuna best:')
        print(json.dumps(metrics, indent=2))
        trades_df.to_csv(Path(out_dir) / 'best_optuna_trades.csv', index=False)
        equity_df.to_csv(Path(out_dir) / 'best_optuna_equity.csv')
        return

    if args.run_grid:
        print('Starting grid/random search — this may take a long time depending on grid_sample and model cost.')
        df_results, best_config = run_grid_search(df.copy(), args, out_dir, sample_size=args.grid_sample, seed=args.grid_seed, objective=args.grid_objective)
        print('Best config saved to', Path(out_dir) / 'best_config.json')
        return

    if args.run_best:
        best_path = Path(args.output_dir) / 'best_config.json'
        optuna_best = Path(args.output_dir) / 'optuna_best_config.json'
        if optuna_best.exists():
            with open(optuna_best, 'r') as fh:
                data = json.load(fh)
                best = data.get('best_params', {})
        elif best_path.exists():
            with open(best_path, 'r') as fh:
                best = json.load(fh)
        else:
            print('No best_config.json or optuna_best_config.json found in output_dir. Exiting.')
            return
        print('Loaded best config:', best)
        for k, v in best.items():
            setattr(args, k, v)

    trades_df, equity_df, final_capital, periods_per_year, bh_final = walk_forward_backtest(df, args)

    returns = equity_df["equity"].pct_change().fillna(0).values
    ann_sharpe = annualized_sharpe(returns, periods_per_year)
    mdd = max_drawdown(equity_df["equity"].values) if not equity_df.empty else 0.0
    total_return = (final_capital / args.initial_capital - 1.0) * 100.0

    print("--- BACKTEST SUMMARY ---")
    print(f"Initial capital: ${args.initial_capital:.2f}")
    print(f"Final capital:   ${final_capital:.2f}")
    print(f"Total return:    {total_return:.2f}%")
    print(f"Ann. Sharpe (est): {ann_sharpe:.3f}")
    print(f"Max Drawdown:    {mdd*100:.2f}%")

    exec_df = trades_df[trades_df['side'] != 'NONE'] if not trades_df.empty else trades_df
    print(f"Number of decision rows: {len(trades_df)}")
    print(f"Executed trades: {len(exec_df)}")
    if len(exec_df):
        wins = (exec_df["net_pnl"] > 0).sum()
        print(f"Winning trades: {wins} / {len(exec_df)} ({wins/len(exec_df)*100:.1f}%)")
        print(f"Avg net PnL per executed trade: ${exec_df['net_pnl'].mean():.2f}")

    if bh_final is not None:
        bh_pct = (bh_final / args.initial_capital - 1.0) * 100.0
        print(f"Buy-and-hold final capital (test period): ${bh_final:.2f} ({bh_pct:.2f}%)")

    trades_csv = out_dir / "trades_log_v7.csv"
    equity_csv = out_dir / "equity_curve_v7.csv"
    trades_df.to_csv(trades_csv, index=False)
    equity_df.to_csv(equity_csv)

    plt.figure(figsize=(10, 5))
    if not equity_df.empty:
        plt.plot(equity_df.index, equity_df["equity"], label="equity")
    plt.title("Paper-trade equity curve (v7)")
    plt.xlabel("time")
    plt.ylabel("USD")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plot_file = out_dir / "equity_curve_v7.png"
    plt.savefig(plot_file)
    plt.show()

    print("Saved files to:", out_dir)
    print("Trades CSV:", trades_csv)
    print("Equity CSV:", equity_csv)


if __name__ == "__main__":
    main()
