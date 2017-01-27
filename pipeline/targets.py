import luigi
import os
import pickle
import shutil
import tensorflow as tf

from abc import ABCMeta, abstractmethod
from multiprocessing.pool import Pool
from pipeline import settings, helpers


class CleanableTarget(luigi.Target, metaclass=ABCMeta):
    """Abstract class of target that can be removed.

    Note:
        Inheritor must define method clean_up.
    """

    @abstractmethod
    def clean_up(self):
        """Method clear all target data"""

        pass


class StorageTarget(CleanableTarget):
    INFO_FILE_NAME = '.storage_info'
    STATUS_FIELD_NAME = '.status'
    STATUS_OK = 'OK'

    def __init__(self, folder_name):
        self.folder_path = os.path.join(settings.DATA_FOLDER_PATH, folder_name)
        helpers.mkdir_if_not_exists(self.folder_path)

        self._info_file_path = os.path.join(self.folder_path, self.INFO_FILE_NAME)
        self._info_file = open(self._info_file_path, '+b')
        self._init_info_file()

    def write_flag(self, key, value):
        """Write key - value pair into info file"""

        info = pickle.load(self._info_file)
        info[key] = value
        pickle.dump(info, self._info_file)

    def read_flag(self, key):
        """Load value for key from info file"""

        info = pickle.load(self._info_file)
        return info.get(key)

    def mark_as_filled(self):
        """Alias for write_flag that save information about storage fill status"""

        self.write_flag(self.STATUS_FIELD_NAME, self.STATUS_OK)

    def exists(self):
        return self._is_filled()

    def clean_up(self):
        shutil.rmtree(self.folder_path)

    def _is_info_file_exists(self):
        return os.path.isfile(self._info_file)

    def _init_info_file(self):
        if not self._is_info_file_exists():
            pickle.dump(dict(), self._info_file)

    def _is_filled(self):
        return self.read_flag(self.STATUS_FIELD_NAME) == self.STATUS_OK


class TFRecordsSampleTarget(StorageTarget):
    """Storage target for TFRecords samples storing"""

    SAMPLE_GROUPS_LABELS_FIELD_NAME = '.sample_groups_labels'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.write_flag(self.SAMPLE_GROUPS_LABELS_FIELD_NAME, dict())

    def get_sample_group_paths(self, label):
        return self._get_sample_groups()[label]

    def add_sample_group_paths(self, label, paths):
        sample_groups_labels = self.get_sample_groups_labels()
        sample_groups_labels[label] = paths
        self._save_sample_groups_labels(sample_groups_labels)

    def remove_sample_group(self, label):
        sample_groups_labels = self.get_sample_groups_labels()
        del sample_groups_labels[label]
        self._save_sample_groups_labels(sample_groups_labels)

    def get_serialized_example(self, label, num_epochs):
        """Method returns serialized example read from tf records

        Args:
            label (str): Name of tfrecords group (e.g. "train").
            num_epochs (int): How many times you need to read that data.
        """

        with tf.name_scope('raw_input_{label}'.format(label=label)):
            tf_records_paths = self.get_sample_group_paths(label)

            filename_queue = tf.train.string_input_producer(
                [tf_records_paths], num_epochs=num_epochs
            )

            reader = tf.TFRecordReader()
            _, serialized_example = reader.read(filename_queue)

            return serialized_example

    def _save_sample_groups(self, *labels):
        assert isinstance(labels, dict)
        self.write_flag(self.SAMPLE_GROUPS_LABELS_FIELD_NAME, labels)

    def _get_sample_groups(self):
        return self.read_flag(self.SAMPLE_GROUPS_LABELS_FIELD_NAME)
