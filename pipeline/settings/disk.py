import os
from pipeline.helpers import mkdir_if_not_exists

MAIN_FOLDER_PATH = "/var/lib/bagpipe"
mkdir_if_not_exists(MAIN_FOLDER_PATH)
DATA_FOLDER_PATH = os.path.join("/var/lib/bagpipe", 'storages')
mkdir_if_not_exists(DATA_FOLDER_PATH)
