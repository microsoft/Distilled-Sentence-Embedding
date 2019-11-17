import logging
import os
import sys
import time
from datetime import datetime

__logger = logging.getLogger(__name__)


def init_console_logging(log_level=logging.INFO):
    __logger.setLevel(log_level)

    ch = logging.StreamHandler(stream=sys.stdout)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    formatter.converter = time.gmtime
    ch.setFormatter(formatter)
    __logger.addHandler(ch)


def init_file_logging(log_file_name_prefix, output_dir, log_level=logging.INFO):
    __logger.setLevel(log_level)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    now_utc_str = datetime.utcnow().strftime("%Y_%m_%d-%H_%M_%S")
    log_file_path = os.path.join(output_dir, f"{log_file_name_prefix}_{now_utc_str}.log")

    ch = logging.FileHandler(log_file_path)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    formatter.converter = time.gmtime
    ch.setFormatter(formatter)
    __logger.addHandler(ch)


def get_default_logger():
    return __logger


def debug(msg, *args, **kwargs):
    if __logger is not None:
        __logger.debug(msg, *args, **kwargs)


def info(msg, *args, **kwargs):
    if __logger is not None:
        __logger.info(msg, *args, **kwargs)


def warn(msg, *args, **kwargs):
    if __logger is not None:
        __logger.warn(msg, *args, **kwargs)


def error(msg, *args, **kwargs):
    if __logger is not None:
        __logger.error(msg, *args, **kwargs)


def exception(msg, *args, exc_info=True, **kwargs):
    if __logger is not None:
        __logger.exception(msg, *args, exc_info=exc_info, **kwargs)
