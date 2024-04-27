import os 
from pathlib import Path


BATCH_SIZE = 32
EPOCH = 20

QA_DATA_DIR = Path.cwd().parent /"dataset" / "qaood"
REAL_DATA_DIR = Path.cwd().parent / "dataset" / "realood"

SUPERCATEGORIES = ['person', 'animal',
                    'vehicle', 'outdoor',  
                    'accessory', 'sports', 
                    'kitchen', 'food',
                    'furniture', 'electronic',
                    'appliance', 'indoor']

