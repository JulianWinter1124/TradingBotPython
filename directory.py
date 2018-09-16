import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def get_absolute_path(path):
    return os.path.join(ROOT_DIR, path)

def file_exists(path):
    return os.path.isfile(path)

def ensure_directory(path):
    directory = os.path.dirname(path)
    print('Trying to create:', directory)
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
            print("created new directory:", directory)
        except OSError as e:
            print('Error creating directory:')
            print(e)
