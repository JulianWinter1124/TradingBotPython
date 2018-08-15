import logging
import time
from multiprocessing import Queue

import numpy as np

from bot_ai.neural import Neural
from util import data_modifier
from util.data_collector_v2 import DataCollector
from util.data_generator import DataGenerator
from util.data_processor import DataProcessor

from matplotlib import pyplot as plt

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

#TODO: put all this somewhere else.
if __name__ == '__main__':
    #logging.getLogger().setLevel(logging.INFO)

    n_in = 20 #The number of timeseriessteps in the past
    n_out = 5 #The number of (additional) future labels
    q = Queue()
    coll = DataCollector(output_filepath='data/pair_data_unmodified.h5', currency_pairs=['USDT_BTC', 'USDT_ETH'],
                         start_dates=[1503446400, 1503446400], end_dates=[9999999999, 9999999999],
                         time_periods=[300, 300], overwrite=True)
    proc = DataProcessor(queue=q, database_filepath='data/pair_data_unmodified.h5', output_filepath='data/finished_data.h5', use_scaling=True, use_indicators=True, n_in=n_in, n_out=n_out)
    n_features = proc.get_number_of_features() #The number of individual features
    coll.start()
    proc.start()
    gen = DataGenerator('data/finished_data.h5')
    time.sleep(60) #Leave some time for the data Processor # TODO: replace with something useful

    generator = gen.create_data_generator('USDT_BTC', batch_size=64, n_in=n_in, n_features=n_features)
    neur = Neural('BTC', overwrite=True, batch_size=64, output_size=1+n_out)
    model = neur.load_or_build_model(n_in=n_in, n_out=n_out, n_features=n_features, layer_units=[30, 20], activation_function='tanh', loss_function='mse') #TODO: put in neural
    #neur.plot_model_to_file('model.png', False, True)

    data, labels = gen.read_data_and_labels_from_finished_data_file('USDT_BTC', n_in=n_in, n_features=n_features)
    split_i = int(len(data)*0.9)
    history = neur.train_model(data[0:split_i, :], labels[0:split_i, :], data[split_i:, :], labels[split_i:, :], epochs=1, shuffle=True, save=False) #'normal' method

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



