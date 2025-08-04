
from config import get_config

cfg = get_config()

from train import train_model

train_model(dict(cfg))