import os
from pipeline.settings import mkdir

MAIN_FOLDER_PATH = "/var/lib/bagpipe"
mkdir(MAIN_FOLDER_PATH)
DATA_FOLDER_PATH = os.path.join("/var/lib/bagpipe", 'data')
mkdir(DATA_FOLDER_PATH)
TFRECORDS_FOLDER_PATH = os.path.join("/var/lib/bagpipe", 'tfrecords')
mkdir(TFRECORDS_FOLDER_PATH)

STATUS_FILE_NAME = '.storage_status'
STATUS_FIELD_NAME = '.status'
STATUS_OK = 'OK'
