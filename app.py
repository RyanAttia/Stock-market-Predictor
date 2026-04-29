import re
import sys
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from data import resolve_ticker, fetch_history
from model import forecast


# ---------------------------------------------------------------------------
# Query parsing
# ---------------------------------------------------------------------------

_TIME_RULES = [
    (r"(\d+)\s*days?",   1),
    (r"(\d+)\s*weeks?",  7),
    (r"(\d+)\s*months?", 30),
    (r"(\d+)\s*years?",  365),
    (r"a\s+week",        7),
    (r"a\s+month",       30),
    (r"tomorrow",        1),
]

_FILLER = [
    "what will", "what's", "tell me about", "tell me", "how will",
    "predict", "forecast", "look up", "show me", "give me",
    "stock", "share", "shares", "price", "worth",
    "going to be", "going to", "be in", "be", "in", "the",
]


def parse_query(text: str) -> tuple[str, int]:
    """Return (company_query, days_ahead) parsed from natural language."""
    days = None
    clean = text

    for pattern, multiplier in _TIME_RULES:
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if m:
            n = int(m.group(1)) if m.lastindex else 1
            days = n * multiplier
            clean = re.sub(pattern, "", clean, flags=re.IGNORECASE)
            break

    if days is None:
        days = 7  # default to 1 week

    for filler in sorted(_FILLER, key=len, reverse=True):
        clean = re.sub(r"\b" + re.escape(filler) + r"\b", " ", clean, flags=re.IGNORECASE)

    clean = re.sub(r"[?\"',.]", " ", clean)
    company = re.sub(r"\s+", " ", clean).strip()
    return company, days


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_forecast(result: dict, ticker: str, company: str) -> None:
    fig, ax = plt.subplots(figsize=(13, 6))

    history = result["history"]
    forecast_df = result["full_forecast"]
    cutoff = history["ds"].iloc[-1]

    # Show last 6 months of history for readability
    recent = history[history["ds"] >= cutoff - pd.Timedelta(days=180)]
    ax.plot(recent["ds"], recent["y"], color="#1f77b4", linewidth=1.5, label="Historical (6 mo)")

    future = forecast_df[forecast_df["ds"] > cutoff]
    ax.plot(future["ds"], future["yhat"], color="#ff7f0e", linestyle="--", linewidth=2, label="Forecast")
    ax.fill_between(
        future["ds"], future["yhat_lower"], future["yhat_upper"],
        alpha=0.25, color="#ff7f0e", label="95% confidence"
    )

    ax.scatter(
        [result["forecast_date"]], [result["predicted_price"]],
        color="red", zorder=5, s=120,
        label=f"Predicted: ${result['predicted_price']:.2f}",
    )

    direction = "▲" if result["trend"] == "up" else "▼"
    ax.set_title(
        f"{company} ({ticker})  —  {result['days_ahead']}-Day Forecast  "
        f"{direction} {result['change_pct']:+.2f}%",
        fontsize=14,
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    ax.grid(True, alpha=0.25)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

import pandas as pd  # noqa: E402 (needed for Timedelta above)


def run(query: str, show_plot: bool = True) -> None:
    company_query, days = parse_query(query)

    if not company_query:
        print("  Couldn't find a company name in that query. Try: 'NVIDIA 3 days'")
        return

    print(f"\n  Looking up '{company_query}' ({days} day horizon)...")
    ticker, company = resolve_ticker(company_query)
    print(f"  Found: {company} ({ticker})")
    print("  Fetching data and training model — this takes ~10 seconds...")

    try:
        df = fetch_history(ticker)
    except ValueError as e:
        print(f"  Error: {e}")
        return

    result = forecast(df, days)

    arrow = "↑" if result["trend"] == "up" else "↓"
    print()
    print(f"  {'─'*48}")
    print(f"  {company} ({ticker})")
    print(f"  Current price :  ${result['current_price']:.2f}")
    print(f"  Forecast date :  {result['forecast_date'].strftime('%a, %b %d %Y')}")
    print(f"  Predicted     :  ${result['predicted_price']:.2f}  {arrow} {result['change_pct']:+.2f}%")
    print(f"  95% range     :  ${result['lower_bound']:.2f} – ${result['upper_bound']:.2f}")
    print(f"  {'─'*48}\n")

    if show_plot:
        plot_forecast(result, ticker, company)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("\nStock Price Predictor  (powered by yfinance + Prophet)")
    print("Examples:")
    print("  NVIDIA in 3 days")
    print("  What will Apple stock be in 2 weeks?")
    print("  Tesla 1 month")
    print("  AAPL 5 days")
    print("Type 'quit' to exit.\n")

    while True:
        try:
            query = input("Ask: ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break
        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            break
        run(query)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run(" ".join(sys.argv[1:]))
    else:
        main()
