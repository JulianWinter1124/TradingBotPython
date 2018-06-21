import datetime
import logging
import time
from multiprocessing import Process, Queue

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from bot_ai import neural
from bot_ai.neural import Neural
from util.data_collector_v2 import DataCollector
from util.data_generator import DataGenerator
from util.data_processor import DataProcessor

def cleaner():
    pass
    # df = pd.read_json(
    #     'https://poloniex.com/public?command=returnChartData&currencyPair=USDT_BTC&start=1457073300&end=9999999999&period=300',
    #     convert_dates=False)
    # df = df.drop('weightedAverage', 1)  # clean
    # df.to_csv('data/BTCUSD300.csv', index=False)
    # df = pd.read_csv('data/BTCUSD300.csv')
    # df = df.drop(columns=['volume', 'quoteVolume'], axis=1)
    # df = df.tail(150000)  # .reset_index(drop=True)
    # scaler = MinMaxScaler(feature_range=(0, 1))
    # scaled_df = pd.DataFrame(scaler.fit_transform(df.values), columns=df.columns)
    # n_hours = 20
    # n_hours_future = 0
    # n_features = len(scaled_df.columns)
    # reframed_df = de.series_to_supervised(scaled_df, n_hours, n_hours_future)
    # reframed_df.drop(reframed_df.columns[np.arange(-n_features + 1, 0)], axis=1, inplace=True)
    # reframed = reframed_df.values
    #
    # split_index = int(0.8 * len(reframed))
    # train, test = reframed[0:split_index, :], reframed[split_index:, :]
    #
    # n_obs = n_hours * n_features
    # label_index = 1
    # train_X, train_y = train[:, :n_obs], train[:, -1]
    # test_X, test_y = test[:, :n_obs], test[:, -1]
    # print(train_X.shape, train_y.shape)
    # plt.plot(df['date'], df['close'], label='actual')
    # plt.legend

    # plt.show()
    #
    # train_X = train_X.reshape((train_X.shape[0], n_hours, n_features))
    # test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))
    # print(train_X)
    # print(train_y)
    #
    # neur = neural.Neural('BTC_USD_15min', overwrite=True, batch_size=16)
    # model = neur.load_or_build_model(n_hours, n_features)
    # history = neur.train_model(train_X, train_y, test_X, test_y, epochs=20)
    #
    # plt.plot(history.history['loss'], label='train')
    # plt.plot(history.history['val_loss'], label='test')
    # plt.legend()
    # plt.show()
    #
    # yhat = neur.predict(test_X)
    # print(yhat)
    # test_y = test_y.reshape((test_y.shape[0], 1))
    # test_X = test_X.reshape((test_X.shape[0], n_hours * n_features))
    # test_X = test_X[:, -n_features:-1]
    # print(pd.DataFrame(test_X))
    # test_pred = np.concatenate((yhat, test_X), axis=1)
    # test_actual = np.concatenate((test_y, test_X), axis=1)
    # print(pd.DataFrame(test_pred, columns=df.columns))
    # print(pd.DataFrame(test_actual, columns=df.columns))
    # test_pred = pd.DataFrame(scaler.inverse_transform(test_pred), columns=df.columns)
    # test_actual = pd.DataFrame(scaler.inverse_transform(test_actual), columns=df.columns)
    # plt.plot(df['date'].iloc[-len(test_X):], test_pred['close'], label='prediciton')
    # plt.plot(df['date'].iloc[-len(test_X):], test_actual['close'], label='actual', linewidth=0.5)
    # plt.legend()
    # plt.show()


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)

    n_in = 20 #TODO: get from processor
    n_out = 2
    q = Queue()
    proc = DataProcessor(queue=q, database_filepath='data/pair_data_unmodified.h5', output_filepath='data/finished_data.h5', use_scaling=True, use_indicators=True, n_in=n_in, n_out=n_out)
    coll = DataCollector('data/', ['USDT_BTC', 'USDT_ETH'], [1503446400, 1503446400], [9999999999, 9999999999],
                         time_periods=[300, 300], overwrite=False)
    coll.start()
    proc.start()

    gen = DataGenerator('data/finished_data.h5')
    time.sleep(30) #TODO: replace with somethin useful

    n_features = proc.get_number_of_features() # TODO: proc.get_features
    generator = gen.generate('USDT_BTC', batch_size=64, n_in=n_in, n_features=n_features)
    neur = Neural('BTC', overwrite=False, units=300, batch_size=64, output_size=1+n_out, activation='sigmoid')
    model = neur.load_or_build_model_2(n_in=n_in, n_out=n_out, n_features=n_features, layer_units=[30, 20, 10], activation_function='linear', loss_function='mse') #TODO: put in neural
    #neur.plot_model_to_file('model.png', True, True)
    #model = neur.load_or_build_model(n_in, n_features)
    #history = neur.train_model_generator(generator, 100000, 10) #generator method
    data, labels = gen.read_data('USDT_BTC', n_in=n_in, n_features=n_features)
    split_i = int(len(data)*0.9)
    history = neur.train_model(data[0:split_i, :], labels[0:split_i, :], data[split_i:, :], labels[split_i:, :], epochs=1, shuffle=True) #'normal' method
    pred = neur.predict(data)
    print('actual\n', labels)
    print('prediction:\n', pred)
    print(labels[split_i:, :].shape, pred.shape)
    concatenated = np.column_stack((data.reshape((data.shape[0], n_in*n_features))[split_i:, -n_features:], pred[:, 0]))
    print(concatenated.shape)
    scaler = q.get()
    print(scaler.inverse_transform(concatenated))
    coll.terminate()
    proc.terminate()


