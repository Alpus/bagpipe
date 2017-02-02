import os
import pickle

class FilePickler:
    """Class hide using of open function and wrap the main pickle methods"""

    def __init__(self, file_path, init_content=None):
        self._file_path = file_path

        if not os.path.isfile(self._file_path):
            if init_content is None:
                raise FileExistsError("File doesn't exists and init_content is None")
            else:
                self.dump(init_content)

    def dump(self, object_):
        file = open(self._file_path, 'wb')
        pickle.dump(object_, file)
        file.close()

    def load(self):
        file = open(self._file_path, 'rb')
        object_ = pickle.load(file)
        file.close()
        return object_
