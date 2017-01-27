import os

def mkdir_if_not_exists(path):
    if os.path.exists(path):
        if not os.path.isdir(path):
            raise FileExistsError("Path is occupied by some file. Can't create directory")
    else:
        os.makedirs(path)
