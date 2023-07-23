STOCK_DATA_SAVE_DIR = "datasets"
TRAINED_MODEL_DIR = "trained_models"
TENSORBOARD_LOG_DIR = "tensorboard_log"
RESULTS_DIR = "results"
MONITOR_DIR = "monitor_log"
MODEL_NAME = "ppo"

TICKERS = [
    "TCS.NS",
    "BAJFINANCE.NS",
    "BAJAJFINSV.NS",
    "BHARTIARTL.NS",
    "EICHERMOT.NS",
]


INTERVAL = "1h"
PERIOD = "360d"
TRAIN_TEST_SPLIT_PERCENT = 0.15
TECHNICAL_INDICATORS = [
    "RSI_14",
    "EMA_8",
    "EMA_21",
    # "BBAND_UPPER",
    # "BBAND_MIDDLE",
    # "BBAND_LOWER",
    # "DEMA_21",
    # "DEMA_55",
    # "HT_TRENDLINE",
    "KAMA_30",
    "PAST_1_HOUR",
    "PAST_2_HOUR",
    "PAST_3_HOUR",
    "PAST_4_HOUR",
    "PAST_5_HOUR",
    "PAST_10_HOUR",
    "PAST_24_HOUR",
    "PAST_36_HOUR",
    "PAST_48_HOUR",
]

FILENAME = f"{('-'.join(TICKERS))}-{INTERVAL}-{PERIOD}.parquet"
NUMPY_FILENAME = f"{('-'.join(TICKERS))}-{INTERVAL}-{PERIOD}.npy"
SEED = 1337
