"""config.py
"""

import types
import tensorflow as tf
from datetime import datetime

config = types.SimpleNamespace()

# Subdirectory name for saving trained weights and models
config.SAVE_DIR = 'models'

# Subdirectory name for saving TensorBoard log files
config.LOG_DIR = 'logs'

# Default path to the ImageNet TFRecords dataset files
# config.DEFAULT_DATASET_DIR = os.path.join(os.environ['HOME'], 'data/ILSVRC2012/tfrecords')

# Number of parallel works for generating training/validation data
config.NUM_DATA_WORKERS = 8
#
# Do image data augmentation or not
config.DATA_AUGMENTATION = True

# tf records creation parameters
config.FILE_PATTERN = 'dataset_%s_%05d-of-%05d.tfrecord'
config.LABEL_FILE_NAME = 'dataset_labels.txt'
config.RANDOM_SEED = 10

# tf summary
config.file_writer = None
