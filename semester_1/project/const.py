import os
from pathlib import Path
from torch import device
from torch.cuda import is_available

ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

default_device = device('cuda' if is_available() else 'cpu')