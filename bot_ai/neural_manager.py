import h5py
import multiprocessing as mp

from bot.data_generator import DataGenerator
from bot_ai.neural import Neural


class NeuralManager():
    def __init__(self, finished_data_filepath, overwrite_models, batch_size, output_size, n_in, n_out, n_features, layer_units=[30, 20], activation_function='LeakyReLU', loss_function='LeakyReLU', optimizer='adam'):

        self.finished_data_filepath = finished_data_filepath
        self.overwrite_models = overwrite_models
        self.batch_size = batch_size
        self.output_size = output_size
        self.n_in = n_in
        self.n_out = n_out
        self.n_features = n_features
        self.layer_units = layer_units
        self.activation_function = activation_function
        self.loss_function = loss_function
        self.optimizer = optimizer
        #param end
        self.neural_instances = dict()

    def run(self):
        finished_data_file = self.read_finished_data_file()
        gen = DataGenerator('data/finished_data.hdf5')
        pairs = list(finished_data_file.keys())
        finished_data_file.close()
        res = []
        with mp.Pool(processes=4) as pool:
            pool.close() #res.append(pool.apply_async(func=self.neural_instances[pair].load_or_build_model))
            pool.join()
        for pair in pairs:
            self.neural_instances[pair] = Neural(pair, overwrite=self.overwrite_models, batch_size=self.batch_size,
                                                 output_size=self.output_size, n_in=self.n_in, n_out=self.n_out,
                                                 n_features=self.n_features, layer_units=self.layer_units,
                                                 activation_function=self.activation_function,
                                                 loss_function=self.loss_function, optimizer=self.optimizer)
            neur : Neural = self.neural_instances[pair]
            neur.load_or_build_model()

            generator = gen.create_data_generator(pair, batch_size=self.batch_size, n_in=self.n_in, n_features=self.n_features)
            data, labels = gen.read_data_and_labels_from_finished_data_file('USDT_BTC', n_in=self.n_in,
                                                                            n_features=self.n_features)
            split_i = int(len(data) * 0.9)
            history = neur.train_model(data[0:split_i, :], labels[0:split_i, :], data[split_i:, :], labels[split_i:, :],
                                       epochs=100, shuffle=True, save=True)  # 'normal' method
            pred = neur.predict(data[split_i:, :])
            print('actual\n', labels[split_i:])
            print(labels.shape, pred.shape)
            for i in range(pred.shape[1]):
                print('prediction:' + str(i) + '\n', pred[:, i])
        for result in res:
            result.get()
        #pickling horror incoming?

    def read_finished_data_file(self):
        return h5py.File(self.finished_data_filepath, libver='latest')