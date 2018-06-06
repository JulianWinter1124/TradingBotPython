import h5py


class DataProcessor:

    def __init__(self, database_filepath):
        self.filepath = database_filepath


    def read(self):
        with h5py.File(self.filepath, 'r') as f:
            dset_names = f.keys()
            for key in dset_names:
                dset = f[key]