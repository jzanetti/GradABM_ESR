from datetime import datetime
from logging import INFO, Formatter, StreamHandler, basicConfig, getLogger
from os.path import join
from time import time

from yaml import safe_load as yaml_load


def create_random_seed(factor=100000):
    return factor * time()


def setup_logging(workdir: str = "/tmp", start_utc: datetime = datetime.utcnow()):
    """set up logging system for tasks

    Returns:
        object: a logging object
    """
    formatter = Formatter("%(asctime)s - %(name)s.%(lineno)d - %(levelname)s - %(message)s")
    ch = StreamHandler()
    ch.setLevel(INFO)
    ch.setFormatter(formatter)
    logger_path = join(workdir, f"gradabm_esr.{start_utc.strftime('%Y%m%d')}")
    basicConfig(filename=logger_path),
    logger = getLogger()
    logger.setLevel(INFO)
    logger.addHandler(ch)

    return logger


def read_cfg(cfg_path: str) -> dict:
    """Read configuration file

    Args:
        cfg_path (str): configuration path

    Returns:
        dict: configuration
    """
    with open(cfg_path, "r") as fid:
        cfg = yaml_load(fid)

    return cfg


def round_a_list(input: list, sig_figures: int = 3):
    return [round(x, sig_figures) for x in input]
