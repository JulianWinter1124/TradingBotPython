import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def get_absolute_path(path):
    return os.path.join(ROOT_DIR, path)

def file_exists(path):
    return os.path.isfile(path)