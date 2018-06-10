import datetime
import logging
from multiprocessing import Process

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from bot_ai import neural
from util import data_enhancer as de
from util.data_collector_v2 import DataCollector
from util.data_processor import DataProcessor


def load_data():
    # https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
    df = pd.read_csv('data/BTCUSD300.csv')
    values = df.values
    values = values.astype('float32')
    # normalize features
    print(values.shape)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    scaled_df = pd.DataFrame(scaled)
    scaled_df.columns = df.columns
    # frame as supervised learning
    n_hours = 3
    n_features = len(df.columns)
    reframed = de.series_to_supervised(scaled_df, n_hours, 0)
    # drop columns we don't want to predict
    reframed.drop(reframed.columns[np.arange(-n_features + 1, 0)], axis=1, inplace=True)
    # TODO
    # Making all series stationary with differencing and seasonal adjustment.
    # Providing more than 1 hour of input time steps.
    values = reframed.values
    train_index = int(0.8 * len(values))
    train = values[:train_index, :]
    test = values[train_index:, :]
    # split into input and outputs
    n_obs = n_hours * n_features
    train_X, train_y = train[:, :n_obs], train[:, -n_features]
    test_X, test_y = test[:, :n_obs], test[:, -n_features]
    print(train_X.shape, len(train_X), train_y.shape)
    print(reframed.iloc[:, 0:-1])
    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], n_hours, n_features))
    test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
    print(reframed.head())

    neur = neural.Neural('BTC_USD_15min', overwrite=False)
    model = neur.load_or_build_model(n_hours, n_features)
    history = neur.train_model(train_X, train_y, test_X, test_y, 10)

    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()

    # make a prediction
    yhat = neur.predict(test_X)
    test_X = test_X.reshape((test_X.shape[0], n_hours * n_features))
    inv_yhat = np.concatenate((yhat, test_X[:, -7:]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, 0]
    # invert scaling for actual
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = np.concatenate((test_y, test_X[:, -7:]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    print(inv_y)
    inv_y = inv_y[:, 0]
    print(inv_y)
    # calculate RMSE
    rmse = np.math.sqrt(mean_squared_error(inv_y, inv_yhat))
    print(rmse)
    print(yhat.shape)
    print(test_y.shape)
    print(test_y)

    fig, ax1 = plt.subplots(1, 1)
    ax1.plot(scaler.inverse_transform(test)[:, 1].astype(datetime.datetime), inv_y, label='Actual')
    ax1.plot(scaler.inverse_transform(test)[:, 1].astype(datetime.datetime), inv_yhat, label='Predicted')
    plt.legend()
    plt.show()


def cleaner():
    # df = pd.read_json(
    #     'https://poloniex.com/public?command=returnChartData&currencyPair=USDT_BTC&start=1457073300&end=9999999999&period=300',
    #     convert_dates=False)
    # df = df.drop('weightedAverage', 1)  # clean
    # df.to_csv('data/BTCUSD300.csv', index=False)
    df = pd.read_csv('data/BTCUSD300.csv')
    df = df.drop(columns=['volume', 'quoteVolume'], axis=1)
    df = df.tail(150000)  # .reset_index(drop=True)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_df = pd.DataFrame(scaler.fit_transform(df.values), columns=df.columns)
    n_hours = 20
    n_hours_future = 0
    n_features = len(scaled_df.columns)
    reframed_df = de.series_to_supervised(scaled_df, n_hours, n_hours_future)
    reframed_df.drop(reframed_df.columns[np.arange(-n_features + 1, 0)], axis=1, inplace=True)
    reframed = reframed_df.values

    split_index = int(0.8 * len(reframed))
    train, test = reframed[0:split_index, :], reframed[split_index:, :]

    n_obs = n_hours * n_features
    label_index = 1
    train_X, train_y = train[:, :n_obs], train[:, -1]
    test_X, test_y = test[:, :n_obs], test[:, -1]
    print(train_X.shape, train_y.shape)
    plt.plot(df['date'], df['close'], label='actual')
    plt.legend()
    plt.show()

    train_X = train_X.reshape((train_X.shape[0], n_hours, n_features))
    test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))
    print(train_X)
    print(train_y)

    neur = neural.Neural('BTC_USD_15min', overwrite=True, batch_size=16)
    model = neur.load_or_build_model(n_hours, n_features)
    history = neur.train_model(train_X, train_y, test_X, test_y, epochs=20)

    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()

    yhat = neur.predict(test_X)
    print(yhat)
    test_y = test_y.reshape((test_y.shape[0], 1))
    test_X = test_X.reshape((test_X.shape[0], n_hours * n_features))
    test_X = test_X[:, -n_features:-1]
    print(pd.DataFrame(test_X))
    test_pred = np.concatenate((yhat, test_X), axis=1)
    test_actual = np.concatenate((test_y, test_X), axis=1)
    print(pd.DataFrame(test_pred, columns=df.columns))
    print(pd.DataFrame(test_actual, columns=df.columns))
    test_pred = pd.DataFrame(scaler.inverse_transform(test_pred), columns=df.columns)
    test_actual = pd.DataFrame(scaler.inverse_transform(test_actual), columns=df.columns)
    plt.plot(df['date'].iloc[-len(test_X):], test_pred['close'], label='prediciton')
    plt.plot(df['date'].iloc[-len(test_X):], test_actual['close'], label='actual', linewidth=0.5)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    coll = DataCollector('data', ['USDT_BTC', 'USDT_ETH'], [1405699200, 1405699200], [9999999999, 9999999999],
                         time_periods=[300, 300], overwrite=False)
    proc = DataProcessor(database_filepath=coll.filepath, output_filepath='data/finished_data.hdf5', overwrite=True)
    p1 = Process(target=coll.run_unmodified_loop)
    p2 = Process(target=proc.run)
    p1.start()
    p2.start()
