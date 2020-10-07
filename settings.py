from pathlib import Path
import gzip
import bz2
import torch
import pandas as pd
import pickle
import time

root = Path('.')
rawdata = Path("/mnt/datasets/horseracing")
prepdata = root/"processed"
model = root/"models"
config = root/"config"

prep_data_path = prepdata/'prep_full.pt'
train_data_path = prepdata/'prep_train.pt'
val_data_path = prepdata/'prep_val.pt'


