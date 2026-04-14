from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
CHART_DIR = OUTPUT_DIR / "charts"
TABLE_DIR = OUTPUT_DIR / "tables"


def ensure_dirs() -> None:
    for path in (OUTPUT_DIR, CHART_DIR, TABLE_DIR):
        path.mkdir(parents=True, exist_ok=True)


def sentiment_bucket(label: str) -> str:
    if label in {"Fear", "Extreme Fear"}:
        return "Fear"
    if label in {"Greed", "Extreme Greed"}:
        return "Greed"
    return "Neutral"


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    sentiment = pd.read_csv(DATA_DIR / "sentiment.csv")
    trader = pd.read_csv(DATA_DIR / "trader_data.csv")

    sentiment["date"] = pd.to_datetime(sentiment["date"])
    sentiment["sentiment_bucket"] = sentiment["classification"].map(sentiment_bucket)

    trader["Timestamp IST"] = pd.to_datetime(
        trader["Timestamp IST"], format="%d-%m-%Y %H:%M", errors="coerce"
    )
    trader["trade_date"] = trader["Timestamp IST"].dt.normalize()
    trader["is_buy"] = trader["Side"].eq("BUY").astype(int)
    trader["is_sell"] = trader["Side"].eq("SELL").astype(int)
    trader["is_long"] = trader["Direction"].fillna("").str.contains("Long", case=False).astype(int)
    trader["is_short"] = trader["Direction"].fillna("").str.contains("Short", case=False).astype(int)
    trader["is_profit_trade"] = (trader["Closed PnL"] > 0).astype(int)
    trader["is_loss_trade"] = (trader["Closed PnL"] < 0).astype(int)
    trader["is_realized_trade"] = trader["Closed PnL"].ne(0).astype(int)
    trader["abs_size_usd"] = trader["Size USD"].abs()
    trader["abs_start_position"] = trader["Start Position"].abs()

    return sentiment, trader


def build_data_quality(sentiment: pd.DataFrame, trader: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for name, df in {"sentiment": sentiment, "trader": trader}.items():
        missing = int(df.isna().sum().sum())
        rows.append(
            {
                "dataset": name,
                "rows": int(df.shape[0]),
                "columns": int(df.shape[1]),
                "duplicate_rows": int(df.duplicated().sum()),
                "missing_cells": missing,
                "date_min": str(df["date"].min().date()) if "date" in df.columns else str(df["trade_date"].min().date()),
                "date_max": str(df["date"].max().date()) if "date" in df.columns else str(df["trade_date"].max().date()),
            }
        )
    return pd.DataFrame(rows)


def build_daily_account(sentiment: pd.DataFrame, trader: pd.DataFrame) -> pd.DataFrame:
    # Collapse trade-level rows to one account-day so they can be aligned with daily sentiment.
    daily_account = (
        trader.groupby(["Account", "trade_date"], as_index=False)
        .agg(
            daily_pnl=("Closed PnL", "sum"),
            total_fee=("Fee", "sum"),
            trades=("Trade ID", "count"),
            realized_trades=("is_realized_trade", "sum"),
            profitable_trades=("is_profit_trade", "sum"),
            losing_trades=("is_loss_trade", "sum"),
            avg_trade_size_usd=("abs_size_usd", "mean"),
            median_trade_size_usd=("abs_size_usd", "median"),
            gross_notional_usd=("abs_size_usd", "sum"),
            avg_abs_start_position=("abs_start_position", "mean"),
            buy_trades=("is_buy", "sum"),
            sell_trades=("is_sell", "sum"),
            long_flag_count=("is_long", "sum"),
            short_flag_count=("is_short", "sum"),
        )
    )

    daily_account["trade_win_rate"] = np.where(
        daily_account["realized_trades"] > 0,
        daily_account["profitable_trades"] / daily_account["realized_trades"],
        np.nan,
    )
    daily_account["long_short_ratio"] = np.where(
        daily_account["short_flag_count"] > 0,
        daily_account["long_flag_count"] / daily_account["short_flag_count"],
        np.nan,
    )
    daily_account["net_pnl_after_fees"] = daily_account["daily_pnl"] - daily_account["total_fee"]
    daily_account["active_day"] = 1

    merged = daily_account.merge(
        sentiment[["date", "classification", "value", "sentiment_bucket"]],
        left_on="trade_date",
        right_on="date",
        how="left",
    ).drop(columns=["date"])

    merged["classification"] = merged["classification"].fillna("Unknown")
    merged["sentiment_bucket"] = merged["sentiment_bucket"].fillna("Unknown")

    return merged


def build_account_segments(daily_account: pd.DataFrame) -> pd.DataFrame:
    # Use median-based splits to keep segmentation simple and interpretable.
    account_summary = (
        daily_account.groupby("Account", as_index=False)
        .agg(
            active_days=("active_day", "sum"),
            total_trades=("trades", "sum"),
            avg_daily_trades=("trades", "mean"),
            avg_trade_size_usd=("avg_trade_size_usd", "mean"),
            total_net_pnl=("net_pnl_after_fees", "sum"),
            avg_daily_pnl=("net_pnl_after_fees", "mean"),
            profitable_days=("net_pnl_after_fees", lambda s: (s > 0).sum()),
            losing_days=("net_pnl_after_fees", lambda s: (s < 0).sum()),
            avg_realized_win_rate=("trade_win_rate", "mean"),
        )
    )

    account_summary["profitable_day_ratio"] = (
        account_summary["profitable_days"] / account_summary["active_days"]
    )

    activity_threshold = account_summary["avg_daily_trades"].median()
    size_threshold = account_summary["avg_trade_size_usd"].median()
    consistency_threshold = account_summary["profitable_day_ratio"].median()

    account_summary["activity_segment"] = np.where(
        account_summary["avg_daily_trades"] >= activity_threshold,
        "High activity",
        "Low activity",
    )
    account_summary["size_segment"] = np.where(
        account_summary["avg_trade_size_usd"] >= size_threshold,
        "Large size",
        "Small size",
    )
    account_summary["consistency_segment"] = np.where(
        account_summary["profitable_day_ratio"] >= consistency_threshold,
        "Consistent winners",
        "Inconsistent",
    )

    return account_summary


def sentiment_performance_table(daily_account: pd.DataFrame) -> pd.DataFrame:
    focus = daily_account[daily_account["sentiment_bucket"].isin(["Fear", "Greed", "Neutral"])].copy()
    table = (
        focus.groupby("sentiment_bucket", as_index=False)
        .agg(
            account_days=("Account", "count"),
            total_net_pnl=("net_pnl_after_fees", "sum"),
            avg_daily_pnl=("net_pnl_after_fees", "mean"),
            median_daily_pnl=("net_pnl_after_fees", "median"),
            daily_win_rate=("net_pnl_after_fees", lambda s: (s > 0).mean()),
            avg_trade_win_rate=("trade_win_rate", "mean"),
            avg_trades=("trades", "mean"),
            avg_trade_size_usd=("avg_trade_size_usd", "mean"),
            avg_gross_notional_usd=("gross_notional_usd", "mean"),
            avg_fee=("total_fee", "mean"),
            avg_abs_start_position=("avg_abs_start_position", "mean"),
        )
        .sort_values("sentiment_bucket")
    )
    return table


def segment_performance_table(
    daily_account: pd.DataFrame, account_segments: pd.DataFrame
) -> pd.DataFrame:
    enriched = daily_account.merge(
        account_segments[
            ["Account", "activity_segment", "size_segment", "consistency_segment"]
        ],
        on="Account",
        how="left",
    )

    # Reshape segment labels into one long table so every segment family is summarized the same way.
    melted = enriched.melt(
        id_vars=["Account", "trade_date", "net_pnl_after_fees", "trades", "sentiment_bucket"],
        value_vars=["activity_segment", "size_segment", "consistency_segment"],
        var_name="segment_type",
        value_name="segment_label",
    )
    melted = melted[melted["sentiment_bucket"].isin(["Fear", "Greed", "Neutral"])].copy()

    table = (
        melted.groupby(["segment_type", "segment_label", "sentiment_bucket"], as_index=False)
        .agg(
            account_days=("Account", "count"),
            avg_net_pnl=("net_pnl_after_fees", "mean"),
            median_net_pnl=("net_pnl_after_fees", "median"),
            avg_trades=("trades", "mean"),
            positive_day_rate=("net_pnl_after_fees", lambda s: (s > 0).mean()),
        )
        .sort_values(["segment_type", "segment_label", "sentiment_bucket"])
    )
    return table


def top_accounts_table(account_segments: pd.DataFrame) -> pd.DataFrame:
    return account_segments.sort_values("total_net_pnl", ascending=False).head(10)


def save_table(df: pd.DataFrame, name: str) -> None:
    df.to_csv(TABLE_DIR / f"{name}.csv", index=False)


def make_charts(
    daily_account: pd.DataFrame, perf_table: pd.DataFrame, segment_table: pd.DataFrame
) -> None:
    sns.set_theme(style="whitegrid", palette="Set2")
    focus = daily_account[daily_account["sentiment_bucket"].isin(["Fear", "Greed", "Neutral"])].copy()

    plt.figure(figsize=(9, 5))
    sns.boxplot(data=focus, x="sentiment_bucket", y="net_pnl_after_fees", showfliers=False)
    plt.title("Daily Net PnL by Sentiment")
    plt.xlabel("Sentiment Bucket")
    plt.ylabel("Net PnL After Fees")
    plt.tight_layout()
    plt.savefig(CHART_DIR / "daily_pnl_by_sentiment.png", dpi=200)
    plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    metrics = [
        ("avg_trades", "Avg Trades / Account-Day"),
        ("avg_trade_size_usd", "Avg Trade Size (USD)"),
        ("daily_win_rate", "Positive Day Rate"),
    ]
    for ax, (column, label) in zip(axes, metrics):
        sns.barplot(data=perf_table, x="sentiment_bucket", y=column, ax=ax)
        ax.set_title(label)
        ax.set_xlabel("")
        ax.set_ylabel(label)
    fig.suptitle("Behavior Shift Across Sentiment Regimes", y=1.03)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "behavior_shift_by_sentiment.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    heatmap_source = segment_table[
        segment_table["segment_type"].eq("activity_segment")
    ].pivot(index="segment_label", columns="sentiment_bucket", values="avg_net_pnl")
    plt.figure(figsize=(8, 4))
    sns.heatmap(heatmap_source, annot=True, fmt=".1f", cmap="RdYlGn", center=0)
    plt.title("Average Net PnL by Activity Segment and Sentiment")
    plt.xlabel("Sentiment Bucket")
    plt.ylabel("Activity Segment")
    plt.tight_layout()
    plt.savefig(CHART_DIR / "activity_segment_heatmap.png", dpi=200)
    plt.close()

    plt.figure(figsize=(9, 5))
    sentiment_order = ["Fear", "Neutral", "Greed"]
    plot_data = focus.groupby(["trade_date", "sentiment_bucket"], as_index=False)["net_pnl_after_fees"].sum()
    for bucket in sentiment_order:
        subset = plot_data[plot_data["sentiment_bucket"].eq(bucket)].sort_values("trade_date")
        subset = subset.assign(cumulative_pnl=subset["net_pnl_after_fees"].cumsum())
        plt.plot(subset["trade_date"], subset["cumulative_pnl"], label=bucket, linewidth=2)
    plt.title("Cumulative Net PnL by Sentiment Bucket")
    plt.xlabel("Trade Date")
    plt.ylabel("Cumulative Net PnL")
    plt.legend()
    plt.tight_layout()
    plt.savefig(CHART_DIR / "cumulative_pnl_by_sentiment.png", dpi=200)
    plt.close()


def build_key_findings(
    perf_table: pd.DataFrame, segment_table: pd.DataFrame, top_accounts: pd.DataFrame
) -> pd.DataFrame:
    perf_idx = perf_table.set_index("sentiment_bucket")
    fear = perf_idx.loc["Fear"]
    greed = perf_idx.loc["Greed"]
    neutral = perf_idx.loc["Neutral"]

    activity = segment_table[segment_table["segment_type"].eq("activity_segment")].copy()
    high_activity = activity[activity["segment_label"].eq("High activity")].set_index("sentiment_bucket")
    low_activity = activity[activity["segment_label"].eq("Low activity")].set_index("sentiment_bucket")

    pnl_gap = float(fear["avg_daily_pnl"] - greed["avg_daily_pnl"])
    trade_gap = float(greed["avg_trades"] - fear["avg_trades"])
    size_gap = float(greed["avg_trade_size_usd"] - fear["avg_trade_size_usd"])

    rows = [
        {
            "finding": "Average net PnL gap: Fear minus Greed",
            "value": round(pnl_gap, 2),
            "unit": "USD per account-day",
        },
        {
            "finding": "Trade frequency gap: Greed minus Fear",
            "value": round(trade_gap, 2),
            "unit": "trades per account-day",
        },
        {
            "finding": "Average trade size gap: Greed minus Fear",
            "value": round(size_gap, 2),
            "unit": "USD",
        },
        {
            "finding": "High activity traders beat low activity traders on Fear days",
            "value": round(
                float(high_activity.loc["Fear", "avg_net_pnl"] - low_activity.loc["Fear", "avg_net_pnl"]),
                2,
            ),
            "unit": "USD per account-day",
        },
        {
            "finding": "Top account total net PnL",
            "value": round(float(top_accounts.iloc[0]["total_net_pnl"]), 2),
            "unit": "USD",
        },
        {
            "finding": "Positive-day rate gap: Greed minus Fear",
            "value": round(float(greed["daily_win_rate"] - fear["daily_win_rate"]), 4),
            "unit": "share",
        },
    ]
    return pd.DataFrame(rows)


def main() -> None:
    ensure_dirs()
    raw_sentiment = pd.read_csv(DATA_DIR / "sentiment.csv")
    raw_trader = pd.read_csv(DATA_DIR / "trader_data.csv")
    sentiment, trader = load_data()

    sentiment_shape = raw_sentiment.shape
    trader_shape = raw_trader.shape
    raw_sentiment["date"] = pd.to_datetime(raw_sentiment["date"])
    raw_trader["Timestamp IST"] = pd.to_datetime(
        raw_trader["Timestamp IST"], format="%d-%m-%Y %H:%M", errors="coerce"
    )
    raw_trader["trade_date"] = raw_trader["Timestamp IST"].dt.normalize()

    data_quality = build_data_quality(raw_sentiment, raw_trader)
    data_quality.loc[data_quality["dataset"].eq("sentiment"), "columns"] = sentiment_shape[1]
    data_quality.loc[data_quality["dataset"].eq("trader"), "columns"] = trader_shape[1]
    daily_account = build_daily_account(sentiment, trader)
    account_segments = build_account_segments(daily_account)
    perf_table = sentiment_performance_table(daily_account)
    segment_table = segment_performance_table(daily_account, account_segments)
    top_accounts = top_accounts_table(account_segments)
    findings = build_key_findings(perf_table, segment_table, top_accounts)

    save_table(data_quality, "data_quality")
    save_table(daily_account, "daily_account_metrics")
    save_table(account_segments, "account_segments")
    save_table(perf_table, "sentiment_performance")
    save_table(segment_table, "segment_performance")
    save_table(top_accounts, "top_accounts")
    save_table(findings, "key_findings")

    make_charts(daily_account, perf_table, segment_table)

    print("Analysis complete.")
    print("\nData quality")
    print(data_quality.to_string(index=False))
    print("\nSentiment performance")
    print(perf_table.to_string(index=False))
    print("\nKey findings")
    print(findings.to_string(index=False))


if __name__ == "__main__":
    main()
