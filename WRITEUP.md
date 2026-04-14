# Trader Performance vs Market Sentiment

## Methodology

I merged daily Bitcoin sentiment with Hyperliquid trading activity at the account-day level. The analysis uses net PnL after fees, trade win rate, trade frequency, average trade size, and gross notional as the core behavioral and performance metrics. Sentiment was grouped into `Fear`, `Greed`, and `Neutral` buckets to keep the comparison easy to interpret.

I also segmented accounts into three simple archetypes using medians from the sample: `High activity` vs `Low activity`, `Large size` vs `Small size`, and `Consistent winners` vs `Inconsistent`. This makes it easier to turn the patterns into actionable rules instead of reporting only overall averages.

## Main insights

1. Fear days generated higher average net PnL per account-day than Greed days: about `$5.04k` vs `$4.07k`, a gap of roughly `$970`.
2. Greed days were more consistent, even though they produced lower average PnL. Positive-day rate was `63.5%` on Greed days vs `59.2%` on Fear days, and median daily PnL was also higher on Greed days (`$236` vs `$105`).
3. Trader behavior became more aggressive on Fear days, not on Greed days. Average trades per account-day were `105.4` on Fear vs `76.9` on Greed, and average trade size was about `$8.53k` on Fear vs `$5.95k` on Greed.
4. High-activity traders were the strongest segment in stressed conditions. On Fear days, high-activity accounts averaged about `$7.70k` net PnL per day compared with `$2.48k` for low-activity accounts.
5. Large-size traders were especially weak on Greed days. Their median net PnL on Greed days was slightly negative, while small-size traders had much healthier Greed-day performance and a `72.6%` positive-day rate.
6. Consistent winners made more money on Fear days, but the highest Greed-day upside came from inconsistent traders. That suggests Greed rewards selective risk-taking, while Fear rewards operational discipline.

## Strategy ideas

1. On Fear days, allow higher activity only for proven active traders, because this segment captured the biggest upside. For low-activity accounts, keep size constrained rather than trying to force more trades.
2. On Greed days, reduce position size for large-size traders and favor smaller, more selective trades. The data suggests Greed produces steadier but smaller gains, while oversized risk hurts median outcomes.

## Deliverables generated

- Data quality summary: `outputs/tables/data_quality.csv`
- Sentiment comparison table: `outputs/tables/sentiment_performance.csv`
- Segment comparison table: `outputs/tables/segment_performance.csv`
- Top accounts table: `outputs/tables/top_accounts.csv`
- Supporting charts in `outputs/charts/`
