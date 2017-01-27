import os
import pickle
from pipeline import DataProcessor, TFRecordsMaker, TFModel

class NumbersMaker(DataProcessor):
    def process_data(self, folder_path):
        filenames = [
            os.path.join(folder_path, 'sample_{num}'.format(num=num))
            for num in range(2)
        ]
        pickle.dump(1, filenames[0])
        pickle.dump(2, filenames[1])

class NumbersAugmentator(DataProcessor):
    def requires(self):
        return NumbersMaker(force=True)

    def process_data(self, folder_path):
        filenames = [
            os.path.join(folder_path, 'sample_{num}'.format(num=num))
            for num in range(2)
        ]
        pickle.dump(1, filenames[0])
        pickle.dump(2, filenames[1])


class NumberTFRecordsMaker(TFRecordsMaker):
    def requires(self):
        return NumberFilesMaker(force=True)

    def get_labeled_file_groups(self):
        """Method must return dict with labeled sample files.

        Example:
            return {
                'train': [(img_1, mask_1), (img_2, mask_2)],
                'test': [(img_3, mask_3)],
            }
        """

        number_files_maker = self.requires()
        return {

        }

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