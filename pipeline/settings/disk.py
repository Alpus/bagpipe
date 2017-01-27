import os
from pipeline.helpers import mkdir_if_not_exists

MAIN_FOLDER_PATH = "/var/lib/bagpipe"
mkdir_if_not_exists(MAIN_FOLDER_PATH)
DATA_FOLDER_PATH = os.path.join("/var/lib/bagpipe", 'data')
mkdir_if_not_exists(DATA_FOLDER_PATH)
TFRECORDS_FOLDER_PATH = os.path.join("/var/lib/bagpipe", 'tfrecords')
mkdir_if_not_exists(TFRECORDS_FOLDER_PATH)
