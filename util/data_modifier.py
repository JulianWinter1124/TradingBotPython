import numpy as np
import pandas as pd
import talib #windows: https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def create_test_and_training_data(normalized_data, label_indices=[0], del_columns_indices=[], split_factor=0.8):
    normalized_data = np.delete(normalized_data, del_columns_indices, axis=1)
    labels = normalized_data[:, label_indices]
    train_test = np.delete(normalized_data, label_indices, axis=1)

    # split and reshape for ML
    split_index = int(split_factor * len(train_test))
    train = train_test[:split_index, :]
    test = train_test[split_index:, :]
    train_label = labels[:split_index, :]
    test_label = labels[split_index:, :]
    train = train.reshape((train.shape[0], train.shape[1], 1))
    test = test.reshape((test.shape[0], test.shape[1], 1))
    train_label = train_label.reshape((train_label.shape[0], train_label.shape[1], 1))
    test_label = test_label.reshape((test_label.shape[0], test_label.shape[1], 1))
    return train, train_label, test, test_label


def normalize_data_MinMax(data):
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    return data, scaler


def normalize_data_Standard(data):
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    return data, scaler


def normalize_data(data, scaler):
    if (scaler != None):
        return scaler.transform(data)
    else:
        return data


def data_to_supervised(data, n_in=1, n_out=1, dropnan=True, drop_columns_indices=[], label_columns_indices=[0]):
    data = np.delete(data, drop_columns_indices, axis=1)
    cols, names = list(), list()
    for i in range(n_in, -n_out - 1, -1):
        column = df.shift(i)
        if i <= 0:
            column = column[:, label_columns_indices]
        cols.append(column)
    # put it all together
    agg = np.concatenate(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def add_SMA_indicator_to_data(data, close_index=0, timeperiod=30):
    out = np.expand_dims(talib.SMA(data[:, close_index], timeperiod=timeperiod), axis=1)
    return np.concatenate([data, out], axis=1)


def add_RSI_indicator_to_data(data, close_index=0, timeperiod=14):
    out = np.expand_dims(talib.RSI(data[:, close_index], timeperiod=timeperiod), axis=1)
    return np.concatenate([data, out], axis=1)


def add_BBANDS_indicator_to_data(data, close_index=0, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0):
    upperband, middleband, lowerband = talib.BBANDS(data[:, close_index], timeperiod=timeperiod, nbdevup=nbdevup,
                                                    nbdevdn=nbdevdn, matype=matype)
    list = (data, np.expand_dims(upperband, axis=1), np.expand_dims(middleband, axis=1), np.expand_dims(lowerband, axis=1))
    return np.concatenate(list, axis=1)


def add_OBV_indicator_to_data(data, close_and_volume_index=[0, 5]):  # Volume indicator
    out = np.expand_dims(talib.OBV(data[:, close_and_volume_index[0]], data[:, close_and_volume_index[1]]), axis=1)
    return np.concatenate((data, out), axis=1)

def drop_NaN_rows(data):
    return data[~np.isnan(data).any(axis=1)]

def create_binary_labels(closing_price_column):
    """Calculate labels (-1 for down or 1 for up)
    :param closing_price_column: The data the labels are generated from
    :return: the labels in a list with length
    """
    label_list = [np.sign(closing_price_column[i] - closing_price_column[i - 1]) for i in
                  range(1, len(closing_price_column))]
    return label_list


def create_ranged_labels(closing_price_column):
    """
    :param closing_price_column: The data the labels are generated from
    :return: difference between prices unified to (-1, 1) in R
    """
    high = np.max(closing_price_column)
    low = np.min(closing_price_column)  # Bei groesser werdenden Daten evtl. aendern
    label_list = [(closing_price_column[i] - closing_price_column[i - 1]) / (high - low) for i in
                  range(1, len(closing_price_column))]
    return label_list
