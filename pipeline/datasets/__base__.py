import glob
import luigi
import numpy as np
import os
import pickle
from abc import ABCMeta, abstractmethod
from pipeline import settings


class SampleObject(metaclass=ABCMeta):
    """Abstract class of sample object

    Note:
        User must define method "make_tf_tensor".
    """

    @abstractmethod
    def as_tf_tensor(self):
        """Method must transform object to tf tensor"""

        pass


class DatasetSummary:
    def __init__(self, length):
        self.length = length


class Dataset(luigi.Task, metaclass=ABCMeta):
    """Class for dataset making and using.

    Note:
        User must define method "make_sample_objects".
        Besides that, class might include method "requires".

    Todo:
        * Write methods for saving data on disk
    """

    @abstractmethod
    def make_sample_objects(self):
        """Method must be a generator that yields SampleObject instances."""

        yield None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._dataset_folder = os.path.join(settings.DATASETS_FOLDER, self.__class__.__name__)
        self._dataset_summary_file = os.path.join(self._dataset_folder, settings.SAMPLE_SUMMARY_FILE_NAME)
        self._sample_length = 0

    def _save_dataset_summary(self):
        file = open(self._dataset_summary_file, 'wb')
        pickle.dump(DatasetSummary(length=self._sample_length), file)

    def _load_dataset_summary(self):
        try:
            file = open(self._dataset_summary_file, 'rb')
        except:
            return None
        return pickle.load(file)

    def _get_sample_objects_paths_list(self, shuffle_seed=None):
        """Returns list of paths to all sample objects"""

        sample_objects_paths = sorted(
            [path for path in glob.glob(self._dataset_folder) if path.endswith(settings.SAMPLE_FILE_SUFFIX)]
        )
        if shuffle_seed is not None:
            rand = np.random.RandomState(shuffle_seed)
            rand.shuffle(sample_objects_paths)
        return sample_objects_paths

    def complete(self):
        """Method returns true if sample is created and saved"""

        dataset_summary = self._load_dataset_summary()
        sample_object_paths = self._get_sample_objects_paths_list()
        return dataset_summary is not None and dataset_summary.length == len(sample_object_paths)

    def _make_sample(self):
        """Save sample on disk"""

        for number, sample_object in enumerate(self.make_sample_objects()):
            assert isinstance(sample_object, SampleObject),\
                f'Method "make_sample_objects" must yield SampleObject instances.'
            file = open(os.path.join(self._dataset_folder, str(number) + settings.SAMPLE_FILE_SUFFIX), 'wb')
            pickle.dump(sample_object, file)
            self._sample_length += 1

        self._save_dataset_summary()

    def get_all_sample(self, shuffle_seed=42):
        """Generator that yelds sample from disk.

        Args:
            shuffle_seed (:obj:`int`, optional): Random sid for sample suffling. Defaults to None.
                If None sample will not be shuffled.
        """

        for path in self._get_sample_objects_paths_list(shuffle_seed):
            yield pickle.load(path)

    def get_tf_train_validation_test(self, train_rate, validation_rate, test_rate, shuffle_seed=42):
        """Returns 3 generators with train, validation and test sample using rates.

        Args:
            train_rate(float): rate of train subsample
            validation_rate(float): rate of validation subsample
            test_rate(float): rate of test subsample
            shuffle_seed (:obj:`int`, optional): Random sid for sample suffling. Defaults to None.
                If None sample will not be shuffled.
        """

        rates = train_rate, validation_rate, test_rate
        assert sum(rates) == 1, "Sum of rates need to be equals 1."

        main_generator = self.get_all_sample(shuffle_seed=shuffle_seed)
        def sample_generator(begin, end):
            for number, sample_object in enumerate(main_generator):
                if number == end:
                    break
                if number >= begin:
                    yield sample_object.as_tf_tensor()

        batch_borders = [border * self._sample_length for border in [0, train_rate, train_rate + validation_rate, 1]]
        borders_list = [
            (0, train_rate),
            (train_rate, train_rate + validation_rate),
            (train_rate + validation_rate, 1)
        ]
        return [sample_generator(begin, end) for begin, end in borders_list]

    def run(self):
        self._make_sample()
