import random
import torch
import os
from torch.utils.data import Dataset
import numpy as np
from .spaces_dataset.code.utils import ReadScene
from PIL import Image
from glob import glob
import torch.nn.functional as F


class NVSDataset(Dataset):
    def __init__(self,
                 data_folder,
                 mode,
                 n_views,
                 n_depths,
                 max_h=256,
                 max_w=448,
                 depth_min=0.5,
                 depth_max=100):
        super().__init__()

        