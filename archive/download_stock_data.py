from config import TICKERS, STOCK_DATA_SAVE_DIR, PERIOD, INTERVAL, FILENAME
from pathlib import Path
from logger_config import download_data_logger as log

import yfinance as yf


def download_stock_data():
    df = yf.download(
        tickers=TICKERS,
        period=PERIOD,
        interval=INTERVAL,
        group_by="Ticker",
        auto_adjust=True,
        prepost=True,
    )
    df = df.stack(level=0).rename_axis(["Date", "Ticker"]).reset_index(level=1)
    df = df.reset_index()
    log.info(f"Downloaded data - \n{df.head(10).to_markdown()}")
    dir = Path(STOCK_DATA_SAVE_DIR)
    dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dir / f"random-{len(TICKERS)}-tickers.parquet", engine="fastparquet")
    log.info(f"Wrote to to file - {dir/FILENAME}")


if __name__ == "__main__":
    download_stock_data()
