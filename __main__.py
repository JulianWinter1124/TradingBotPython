import datetime
import math

import pandas as pd

from bot_ai import neural
from util import data_enhancer as de
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import numpy as np


def load_data():
    #https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
    df = pd.read_csv('data/BTCUSD300.csv')
    values = df.values
    values = values.astype('float32')
    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    scaled_df = pd.DataFrame(scaled)
    scaled_df.columns = df.columns
    # frame as supervised learning
    n_hours = 3
    n_features = len(df.columns)
    reframed = de.series_to_supervised(scaled_df, n_hours, 0)
    # drop columns we don't want to predict
    reframed.drop(reframed.columns[np.arange(-n_features+1, 0)], axis=1, inplace=True)
    # TODO
    # One-hot encoding wind speed.
    # Making all series stationary with differencing and seasonal adjustment.
    # Providing more than 1 hour of input time steps.
    values = reframed.values
    train_index = int(0.8*len(values))
    train = values[:train_index, :]
    test = values[train_index:, :]
    # split into input and outputs
    n_obs = n_hours * n_features
    train_X, train_y = train[:, :n_obs], train[:, -n_features]
    test_X, test_y = test[:, :n_obs], test[:, -n_features]
    print(train_X.shape, len(train_X), train_y.shape)
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
    print(yhat.shape)
    print(test_y.shape)
    print(test_y)

    fig, ax1 = plt.subplots(1, 1)
    ax1.plot(scaler.inverse_transform(test)[:, 1].astype(datetime.datetime), inv_y, label='Actual')
    ax1.plot(scaler.inverse_transform(test)[:, 1].astype(datetime.datetime), inv_yhat, label='Predicted')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    #df = pd.read_json('https://poloniex.com/public?command=returnChartData&currencyPair=USDT_BTC&start=1405699200&end=9999999999&period=300', convert_dates=False)
    #df.to_csv('data/BTCUSD300.csv', index=False)
    #df = pd.read_csv('data/BTCUSD300.csv')
    # print(type(df.dtypes))
    # print(df.head(5))
    # print(de.series_to_supervised(df, 1, 2).head(5))
    # data = df.values
    # train, test = de.split_dataset_in_training_and_test(data, 0.80)
    #
    # train_X, train_y = de.create_shifted_datasets(train)
    # test_X, test_y = de.create_shifted_datasets(test)
    #
    # prices = data[:, 0]  # Get the close price column
    # train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    # test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    # print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
    #
    # neur = neural.Neural()
    # model = neur.build_model(train_X)
    # history = neur.train_model(model, train_X, train_y, test_X, test_y)
    #
    # plt.plot(history.history['loss'], label='train')  # OmegaLUL
    # plt.plot(history.history['val_loss'], label='test')
    # plt.legend()
    # plt.show()
    #
    # lul = model.predict(test_X)
    # print(lul)
    load_data()
