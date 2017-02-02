import luigi
import os
import pickle

import numpy as np
import tensorflow as tf

from pipeline import DataProcessor, TFRecordsMaker, TFModel


class NumbersMaker(DataProcessor):
    def process_data(self, folder_path):
        filenames = [
            os.path.join(folder_path, 'sample_{num}'.format(num=num))
            for num in range(2)
        ]

        files = [open(filename, 'wb') for filename in filenames]

        pickle.dump(np.array([1, 2]), files[0])
        pickle.dump(np.array([3, 4]), files[1])

class NumbersAugmentator(DataProcessor):
    def requires(self):
        return NumbersMaker(force=True)

    def process_data(self, folder_path):
        input_folder_path = self.input().data_folder_path

        for input_file_name in os.listdir(input_folder_path):
            input_file_path = os.path.join(input_folder_path, input_file_name)
            input_file = open(input_file_path, 'rb')

            augmented_data = pickle.load(input_file) * 2

            augmented_data_file_path = os.path.join(folder_path, input_file_name)
            augmented_data_file = open(augmented_data_file_path, 'wb')

            pickle.dump(augmented_data, augmented_data_file)


class NumbersTFRecordsMaker(TFRecordsMaker):
    def requires(self):
        return NumbersAugmentator(force=True)

    def make_labeled_file_groups(self):
        input_folder_path = self.input().data_folder_path
        return {
            'group_1': os.listdir(input_folder_path),
            'group_2': os.listdir(input_folder_path),
        }

    def make_tf_example(self, files):
        numbers = pickle.load(files)

        return tf.train.Example(
            features=tf.train.Features(
                {
                    'number_1': tf.train.Feature(
                        int64_list=tf.train.Int64List(value=numbers[0])
                    ),
                    'number_2': tf.train.Feature(
                        int64_list=tf.train.Int64List(value=numbers[1])
                    )
                }
            )
        )


class SumTFModel(TFModel):
    def requires(self):
        return NumbersTFRecordsMaker(force=True, block_size=1)

    def run_model(self):

        serialized_example = self.input().get_serialized_example(label='group_2', num_epochs=1)

        features = tf.parse_single_example(
            serialized_example,
            features={
                'number_1': tf.FixedLenFeature([], tf.int64),
                'number_2': tf.FixedLenFeature([], tf.int64),
            }
        )

        number_1 = features['number_1']
        number_2 = features['number_1']

        a = tf.placeholder(tf.float32)
        b = tf.placeholder(tf.float32)

        c = a + b
        # get the tensorflow session
        sess = tf.Session()

        # initialize all variables
        sess.run(tf.initialize_all_variables())

        # Now you want to sum 2 numbers
        # first set up a dictionary
        # that includes the numbers
        # The key of the dictionary
        # matches the placeholders
        # required for the sum operation
        feed_dict = {a: number_1, b: number_2}

        # now run the sum operation
        ppx = sess.run([c], feed_dict)

        # print the result
        print(ppx)


def test_pipeline():
    luigi.run(main_task_cls=SumTFModel)
