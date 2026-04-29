import yfinance as yf
import pandas as pd


def resolve_ticker(query: str) -> tuple[str, str]:
    """Resolve a company name or ticker string to (ticker, display_name)."""
    query = query.strip()
    try:
        results = yf.Search(query, max_results=1)
        if results.quotes:
            q = results.quotes[0]
            sym = q["symbol"]
            name = q.get("longname") or q.get("shortname") or sym
            return sym, name
    except Exception:
        pass
    sym = query.upper()
    return sym, sym


def fetch_history(ticker: str, years: int = 3) -> pd.DataFrame:
    """Return daily OHLCV history with a tz-naive DatetimeIndex."""
    df = yf.Ticker(ticker).history(period=f"{years}y")
    if df.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'")
    df.index = df.index.tz_localize(None)
    return df
