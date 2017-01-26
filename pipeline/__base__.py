import luigi
import os
import pickle
import tensorflow as tf

from abc import ABCMeta, abstractmethod
from multiprocessing.pool import Pool
from pipeline import settings


class _DataStorage(luigi.Task, metaclass=ABCMeta):
    """Help to save data and check saving status

    Attributes:
        mark_storage_status (function): path to folder where loaded data must be placed
    """

    force = luigi.BoolParameter(default=False, significant=False)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._is_filled_in_runtime = False
        self._storage_folder_path = os.path.join(settings.DATA_FOLDER_PATH, self.__class__.__name__)
        self._storage_status_file = os.path.join(self._storage_folder_path, settings.STATUS_FILE_NAME)

        self._storage_file = open(self._storage_status_file, '+b')
        self._init_storage_file()

    def _is_storage_file_exists(self):
        return os.path.isfile(self._storage_status_file)

    def _init_storage_file(self):
        if not self._is_storage_file_exists():
            pickle.dump(dict(), self._storage_file)

    def save_in_storage(self, key, value):
        storage = pickle.load(self._storage_file)
        storage[key] = value
        pickle.dump(storage, self._storage_file)

    def load_from_storage(self, key):
        storage = pickle.load(self._storage_file)
        return storage.get(key)

    def mark_storage_as_filled(self):
        self.save_in_storage(settings.STATUS_FIELD_NAME, settings.STATUS_OK)
        self._is_filled_in_runtime = True

    def _is_storage_filled(self):
        return (self.force and self._is_filled_in_runtime) or\
               (not self.force and self.load_from_storage(settings.STATUS_FIELD_NAME) == settings.STATUS_OK)

    def complete(self):
        """Method returns true if sample is created and saved"""

        return self._is_storage_filled()


class DataProcessor(_DataStorage):
    """Class for data loading and transformation.

    Note:
        User must define method "process_data".

    Attributes:
        storage_folder_path (str): path to folder where loaded data must be placed
    """

    force = luigi.BoolParameter(default=False, significant=False)

    @abstractmethod
    def process_data(self, folder_path):
        """Method must save data in folder_path."""

        pass

    def run(self):
        self.process_data(self.storage_folder_path)
        self.mark_storage_as_filled()


class TFRecordsMaker(_DataStorage):
    """Class that transform raw data to tfrecords.

    Note:
        User must define methods "get_labeled_file_groups", "make_tf_record" and optional attribute block_size.
    """

    force = luigi.BoolParameter(default=False, significant=False)

    block_size = 500

    def _get_tfrecords_path_storage_key(self, label):
        return 'tfrecords_path_to_{label}'.format(label=label)

    @abstractmethod
    def get_labeled_file_groups(self):
        """Method must return dict with labeled sample files.

        Example:
            return {
                'train': [(img_1, mask_1), (img_2, mask_2)],
                'test': [(img_3, mask_3)],
            }
        """

        pass

    @abstractmethod
    def make_tf_record(self, files):
        """Method must return record that will be passed into TFRecordWriter.write()


        Example:
            ...
            features = tf.train.Features(
                feature={
                    'image': tf.train.Feature(float_list=image_feature),
                }
            )
            example = tf.train.Example(features=features)
            return example.SerializeToString()
        """

        pass

    def _transform_block_to_tf_records(self, block, path_to_save):
        """Save all block in "path_to_save" """

        writer = tf.python_io.TFRecordWriter(path_to_save)
        for files in block:
            record = self.make_tf_record(files)
            writer.write(record)

    def _get_block_file_name(self, label, number):
        return 'block_{label}_{number}.tfrecords'.format(label=label, number=number)

    def _get_blocks_and_paths_to_save(self, label, files):
        return [
            (
                files[number:number + self.block_size],
                os.path.join(self.storage_folder_path, self._get_block_file_name(label, number))
            )
            for number in range(0, len(files), self.block_size)
        ]

    def _write_tf_records(self):
        """Parallel write tf records by groups"""

        labeled_file_groups = self.get_labeled_file_groups()
        for label, file_group in labeled_file_groups.items():
            with Pool() as pool:
                blocks_and_paths_to_save = self._get_blocks_and_paths_to_save(label, file_group)
                pool.starmap(
                    self._transform_block_to_tf_records,
                    blocks_and_paths_to_save
                )
                only_paths_to_save = list(zip(*blocks_and_paths_to_save))[1]
                self.save_in_storage(self._get_tfrecords_path_storage_key(label), only_paths_to_save)

    def get_serialized_example(self, label, num_epochs):
        """Method returns serialized example read from frecords

        
        """
        with tf.name_scope('raw_input_{label}'.format(label=label)):
            tfrecords_paths = self.load_from_storage(
                self._get_tfrecords_path_storage_key(label)
            )

            filename_queue = tf.train.string_input_producer(
                [tfrecords_paths], num_epochs=num_epochs
            )

            reader = tf.TFRecordReader()
            _, serialized_example = reader.read(filename_queue)
            return serialized_example

    def run(self):
        self._write_tf_records()
        self.mark_storage_as_filled()


class TFModel(_DataStorage):
    """Class for model running.

    Note:
        User must define method "run_model".
    """

    @abstractmethod
    def run_model(self):
        pass

    def _write_benchmarks(self):
        pass

    def run(self):
        self.run_model()
        self._write_benchmarks()