import logging, sys
import absl.logging


def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '[%(asctime)s - %(name)s - %(levelname)s] %(message)s')
    handler.setFormatter(formatter)

    logger.handlers = []
    logger.addHandler(handler)

    # Suppress INFO-level logs from absl
    absl.logging.set_verbosity(absl.logging.WARNING)

    return logger