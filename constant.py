import os 
from pathlib import Path


BATCH_SIZE = 32
EPOCH = 20

QA_DATA_DIR = Path.cwd().parent /"dataset" / "qaood"
REAL_DATA_DIR = Path.cwd().parent / "dataset" / "realood"
MELD_DATA_DIR = Path.cwd().parent / "dataset" / "meldood"

SUPERCATEGORIES = ['person', 'animal',
                    'vehicle', 'outdoor',  
                    'accessory', 'sports', 
                    'kitchen', 'food',
                    'furniture', 'electronic',
                    'appliance', 'indoor']

MELD_CATEGORIES = ['surprise', 
                   'anger', 
                   'neutral',
                   'joy',
                   'sadness',
                   'disgust',
                   'fear']