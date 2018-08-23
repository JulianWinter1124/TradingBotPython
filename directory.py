import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def get_absolute_path(path):
    return os.path.join(ROOT_DIR, path)

def file_exists(path):
    return os.path.isfile(path)

def ensure_directory(path):
    directory = os.path.dirname(path)
    print(directory)
    if not os.path.exists(directory):
        try:
            os.makedirs(os.path.dirname(directory))
        except OSError as e:
            print('Error creating directory:')
            print(e)
