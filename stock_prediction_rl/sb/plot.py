import plotly.graph_objects as go
import polars as pl
from pathlib import Path

datasets = Path("datasets")
model_name = "A2C"
num_envs = 16
ticker = "SBIN.NS"


train_file = datasets / f"{ticker}_train"
trade_file = datasets / f"{ticker}_trade"


train_df = pl.read_parquet(train_file)
trade_df = pl.read_parquet(trade_file)

trade_df = trade_df.with_columns(pl.col("Datetime").dt.replace_time_zone(None))
trade_df.write_excel(trade_file.as_posix() + ".xlsx")

# Plotting the close price using plotly
fig = go.Figure(
    data=[
        go.Scatter(
            x=train_df["Datetime"],
            y=train_df["Close"],
            mode="lines",
            name="Close Price",
        )
    ]
)
fig.update_layout(
    title=f"{ticker} Close Price Over Time",
    xaxis_title="Date",
    yaxis_title="Price (in $)",
)

fig.show()
