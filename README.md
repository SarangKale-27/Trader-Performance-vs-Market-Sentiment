# Trader Performance vs Market Sentiment

This repo contains a reproducible analysis of how Bitcoin market sentiment relates to trader behavior and performance on Hyperliquid.

## Project structure

- `analysis.py`: main analysis script
- `data/`: raw assignment datasets
- `outputs/charts/`: generated charts
- `outputs/tables/`: generated summary tables and intermediate outputs
- `WRITEUP.md`: concise methodology, findings, and strategy ideas

## How to run

1. Make sure Python has `pandas`, `numpy`, `matplotlib`, and `seaborn`.
2. Run:

```bash
python analysis.py
```

The script will regenerate all tables and charts inside `outputs/`.

## Method summary

- Loaded both datasets and checked shape, missing values, duplicates, and date overlap.
- Converted trader timestamps to daily granularity and merged each account-day with the corresponding Fear/Greed label.
- Collapsed sentiment into 3 buckets for comparison:
  - `Fear` = Fear + Extreme Fear
  - `Greed` = Greed + Extreme Greed
  - `Neutral` = Neutral
- Built account-day metrics:
  - daily net PnL after fees
  - trade win rate
  - trades per day
  - average trade size
  - gross traded notional
  - long/short activity proxies
- Built account-level segments:
  - high activity vs low activity
  - large size vs small size
  - consistent winners vs inconsistent traders

## Notes and assumptions

- The trader file does not expose a direct `leverage` column, so the analysis uses trade notional and absolute start position as risk proxies.
- Performance is evaluated at the account-day level because that aligns naturally with the daily sentiment series.

## Key output files

- `outputs/tables/data_quality.csv`
- `outputs/tables/sentiment_performance.csv`
- `outputs/tables/segment_performance.csv`
- `outputs/tables/top_accounts.csv`
- `outputs/charts/daily_pnl_by_sentiment.png`
- `outputs/charts/behavior_shift_by_sentiment.png`
- `outputs/charts/activity_segment_heatmap.png`
- `outputs/charts/cumulative_pnl_by_sentiment.png`
