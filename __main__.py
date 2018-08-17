import logging
import time
from multiprocessing import Queue

import numpy as np

from bot_ai.neural import Neural
from util import data_modifier
from util.data_collector_v3 import DataCollector
from util.data_generator import DataGenerator
from util.data_processor import DataProcessor

from matplotlib import pyplot as plt

#TODO: put all this somewhere else.
if __name__ == '__main__':
    #logging.getLogger().setLevel(logging.INFO)

    n_in = 20 #The number of timeseriessteps in the past
    n_out = 5 #The number of (additional) future labels
    q = Queue()
    coll = DataCollector(output_filepath='data/pair_data_unmodified.hdf5', currency_pairs=['USDT_BTC', 'USDT_ETH'],
                         start_dates=[1503446400, 1503446400], end_dates=[9999999999, 9999999999],
                         time_periods=[300, 300], overwrite=True)
    proc = DataProcessor(queue=q, database_filepath='data/pair_data_unmodified.hdf5', output_filepath='data/finished_data.hdf5', use_scaling=True, use_indicators=True, n_in=n_in, n_out=n_out)
    n_features = proc.get_number_of_features() #The number of individual features
    coll.start()
    time.sleep(3)
    proc.start()
    gen = DataGenerator('data/finished_data.hdf5')
    time.sleep(60) #Leave some time for the data Processor # TODO: replace with something useful

    generator = gen.create_data_generator('USDT_BTC', batch_size=64, n_in=n_in, n_features=n_features)
    neur = Neural('BTC', overwrite=True, batch_size=64, output_size=1+n_out)
    model = neur.load_or_build_model(n_in=n_in, n_out=n_out, n_features=n_features, layer_units=[30, 20], activation_function='tanh', loss_function='mse') #TODO: put in neural
    #neur.plot_model_to_file('model.png', False, True)

    data, labels = gen.read_data_and_labels_from_finished_data_file('USDT_BTC', n_in=n_in, n_features=n_features)
    split_i = int(len(data)*0.9)
    history = neur.train_model(data[0:split_i, :], labels[0:split_i, :], data[split_i:, :], labels[split_i:, :], epochs=100, shuffle=True, save=True) #'normal' method

    pred = neur.predict(data[split_i:, :])

    print('actual\n', labels[split_i:])
    print(labels.shape, pred.shape)
    for i in range(pred.shape[1]):
        print('prediction:'+str(i)+'\n', pred[:, i])

    scaler = q.get()
    unscaled = [data_modifier.reverse_normalize_incomplete_data(pred[:, i], 0, n_features, scaler) for i in range(pred.shape[1])] #0 is hopefully the close index
    print(unscaled)

    original_data = gen.read_data_from_database_file('USDT_BTC')
    pred2 = neur.predict(data_modifier.data_to_timeseries_without_labels(original_data, n_in, scaler, use_scaling=True, use_indicators=True))

    plt.figure(figsize=(50,30))
    #plt.plot(original_data[:, 1], original_data[:, 0])
    plt.plot(original_data[split_i+n_in+30:-n_out, 1], original_data[split_i+n_in+30:-n_out, 0], label='actual', linewidth=0.4) #TODO: date conversion
    for i in range(len(unscaled)):
        plt.plot(original_data[split_i+i-1+n_in+30:-n_out+i-1, 1], unscaled[i], label='prediction'+str(i), linewidth=0.4)
    plt.legend()
    plt.show()



