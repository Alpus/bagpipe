import luigi
import os
import tensorflow as tf

from abc import ABCMeta, abstractmethod
from multiprocessing.pool import Pool
from pipeline.targets import CleanableTarget, StorageTarget, TFRecordsSampleTarget


class ForceableTask(luigi.Task, metaclass=ABCMeta):
    """Abstract class of task which running can be forced.

    Note:
        If task forced all output will be removed.
        All output targets must be CleanableTargets.
    """

    force = luigi.BoolParameter(significant=False, default=False)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        flatten_output = luigi.task.flatten(self.output())

        # Check are all output targets instance of CleanableTarget
        if not all(isinstance(target, CleanableTarget) for target in flatten_output):
            raise AttributeError('Not all targets in output() are CleanableTargets.')

        # To force execution, we just remove all outputs before `complete()` is called
        if self.force is True:
            for target in flatten_output:
                target.clean_up()


class DataProcessor(ForceableTask):
    """Class for data loading and transformation.

    Note:
        User must define method "process_data".
    """

    @abstractmethod
    def process_data(self, folder_path):
        """Method must save data in folder_path."""

        pass

    def output(self):
        return StorageTarget(folder_name=self.__class__.__name__)

    def run(self):
        self.process_data(self.output().folder_path)
        self.mark_storage_as_filled()


class TFRecordsMaker(ForceableTask):
    """Class that transform raw data to tfrecords.

    Note:
        User must define methods "get_labeled_file_groups", "make_serialized_tf_record".

    Args:
        block_size (:obj:`int`, optional): sample block size.

    Attributes:
        get_serialized_example (function): Method that returns one serialized example from tfrecords group.
    """

    block_size = luigi.IntParameter(significant=False)

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
    def make_serialized_tf_record(self, files):
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
            record = self.make_serialized_tf_record(files)
            writer.write(record)

    def _get_block_file_name(self, label, number):
        return 'block_{label}_{number}.tfrecords'.format(label=label, number=number)

    def _get_blocks_and_paths_to_save(self, label, files):
        return [
            (
                files[number:number + self.block_size],
                os.path.join(self.storagefolder_path, self._get_block_file_name(label, number))
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

            self.output().add_sample_group_paths(label, only_paths_to_save)

    def output(self):
        return TFRecordsSampleTarget(folder_name=self.__class__.__name__)

    def run(self):
        self._write_tf_records()
        self.mark_storage_as_filled()


class TFModel(ForceableTask):
    """Class for model running.

    Note:
        User must define method "run_model".
    """

    @abstractmethod
    def run_model(self):
        """Method must implement tensoflow model

        Note:
            You probably want to get data using TFRecordsMaker.get_serialized_example(label, num_epochs)
        """

        pass

    def _write_reports(self):
        pass

    def run(self):
        self.run_model()
        self._write_reports()