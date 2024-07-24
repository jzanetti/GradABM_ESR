from process.train_wrapper import train
from process.utils.utils import setup_logging

logger = setup_logging("/tmp/gradabm_esr/test")
train("/tmp/gradabm_esr/test", use_test_data=True)
