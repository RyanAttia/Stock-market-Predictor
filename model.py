import pandas as pd
from prophet import Prophet


def forecast(df: pd.DataFrame, days_ahead: int) -> dict:
    """
    Train a Prophet model on closing prices and predict `days_ahead` business days forward.
    Returns a dict with the key prediction stats and full forecast DataFrame.
    """
    prophet_df = pd.DataFrame({"ds": df.index, "y": df["Close"].values})

    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        interval_width=0.95,
        changepoint_prior_scale=0.05,
    )
    # Suppress Stan output
    import logging
    logging.getLogger("prophet").setLevel(logging.WARNING)
    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

    model.fit(prophet_df)

    future = model.make_future_dataframe(periods=days_ahead, freq="B")
    forecast_df = model.predict(future)

    last_actual = float(df["Close"].iloc[-1])
    predicted_row = forecast_df.iloc[-1]

    predicted = float(predicted_row["yhat"])
    lower = float(predicted_row["yhat_lower"])
    upper = float(predicted_row["yhat_upper"])
    change_pct = (predicted - last_actual) / last_actual * 100

    return {
        "current_price": last_actual,
        "predicted_price": predicted,
        "lower_bound": lower,
        "upper_bound": upper,
        "forecast_date": predicted_row["ds"],
        "days_ahead": days_ahead,
        "trend": "up" if predicted > last_actual else "down",
        "change_pct": change_pct,
        "full_forecast": forecast_df,
        "history": prophet_df,
    }
