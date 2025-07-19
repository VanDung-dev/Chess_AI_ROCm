import logging
from colorlog import ColoredFormatter

def setup_logger():
    logger = logging.getLogger()
    if not logger.handlers:
        # Formatter với màu sắc cho các log
        formatter = ColoredFormatter(
            "%(log_color)s%(levelname)-8s%(reset)s %(white)s%(message)s",
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'bold_red',
            }
        )

        # Thiết lập handler
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)

        # Lấy root logger
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)  # Có thể thay bằng INFO nếu muốn ít log hơn
        logger.addHandler(handler)

    return logger
