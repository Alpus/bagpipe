import os
import pickle
from pipeline import DataProcessor, TFRecordsMaker, TFModel

class MakeNumberFiles(DataProcessor):
    def process_data(self, folder_path):
        filenames = [
            os.path.join(folder_path, 'sample_{num}'.format(num=num))
            for num in range(2)
        ]
        pickle.dump(1, filenames[0])
        pickle.dump(2, filenames[1])

class MakeNumberTFRecords(TFRecordsMaker):
    pass