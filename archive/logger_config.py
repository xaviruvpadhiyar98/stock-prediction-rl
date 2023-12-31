import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
    datefmt="%d/%b/%Y %H:%M:%S",
    stream=sys.stdout,
)
download_data_logger = logging.getLogger("download_data")
add_features_logger = logging.getLogger("add_features")
train_logger = logging.getLogger("train")
eval_logger = logging.getLogger("eval")
