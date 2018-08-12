import logging

import numpy as np
import talib  # windows: https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# def create_test_and_training_data(normalized_data, label_indices=[0], del_columns_indices=[], split_factor=0.8):
#     normalized_data = np.delete(normalized_data, del_columns_indices, axis=1)
#     labels = normalized_data[:, label_indices]
#     train_test = np.delete(normalized_data, label_indices, axis=1)
#
#     # split and reshape for ML
#     split_index = int(split_factor * len(train_test))
#     train = train_test[:split_index, :]
#     test = train_test[split_index:, :]
#     train_label = labels[:split_index, :]
#     test_label = labels[split_index:, :]
#     train = train.reshape((train.shape[0], 1, train.shape[1]))
#     test = test.reshape((test.shape[0], 1, test.shape[1]))
#     train_label = train_label.reshape(
#         (train_label.shape[0], 1, train_label.shape[1]))  # TODO: Change 1 to something else
#     test_label = test_label.reshape((test_label.shape[0], 1, test_label.shape[1]))
#     return train, train_label, test, test_label

def normalize_data_MinMax(data):
    scaler = MinMaxScaler()
    print('scaler fit on data with shape:', data.shape)
    data = scaler.fit_transform(data)
    return data, scaler


def normalize_data_Standard(data):
    scaler = StandardScaler()
    print('scaler fit on data with shape:', data.shape)
    data = scaler.fit_transform(data)
    return data, scaler


def normalize_data(data, scaler):
    if (scaler != None):
        return scaler.transform(data)
    else:
        return data


def reverse_normalize_incomplete_data(data_column, original_index_in_data, n_features, scaler):
    data = np.zeros(shape=(len(data_column), n_features))
    data[:, original_index_in_data] = data_column
    return scaler.inverse_transform(data)[:, original_index_in_data]


def data_to_supervised_timeseries(data, n_in=1, n_out=1, n_out_jumps=1, drop_columns_indices=[], label_columns_indices=[0]):
    """
    Converts the given data to a timeseries. Inspired from https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
    :param data:
    :param n_in: the number of steps you look back in the past (excluding present)
    :param n_out: the number of steps you look in the future (excluding present)
    :param n_out_jumps: the jumps you make in the future (to be able to change timeframe)
    :param drop_columns_indices:
    :param label_columns_indices:
    :return:
    """
    data = np.delete(data, drop_columns_indices, axis=1)
    cols = list()
    for i in range(n_in, -n_out - 1, -1):
        if i <= 0:
            column = np.roll(data, i*n_out_jumps, axis=0)
            column = column[:, label_columns_indices]  # The future points may only contain labels
        else:
            column = np.roll(data, i, axis=0)
        cols.append(column)
    concat = np.concatenate(cols, axis=1)
    if n_out == 0:
        concat = concat[n_in:]# NaN is always dropped
    else:
        concat = concat[n_in:-n_out*n_out_jumps]
    return concat

def data_to_timeseries_without_labels(data, n_in, scaler, drop_columns_indices=[], use_scaling=True, use_indicators=True):
    data = np.delete(data, drop_columns_indices, axis=1)
    selection_array = data[-(n_in+30):, :]
    if use_indicators:  # adding indicators
        selection_array = np.array(selection_array, dtype='f8')
        selection_array = add_SMA_indicator_to_data(selection_array, close_index=0, timeperiod=30)  # 1 column
        selection_array = add_BBANDS_indicator_to_data(selection_array, close_index=0)  # 3 columns
        selection_array = add_RSI_indicator_to_data(selection_array, close_index=0)  # 1 column
        selection_array = add_OBV_indicator_to_data(selection_array, close_index=0, volume_index=6)  # 1column
        selection_array = drop_NaN_rows(selection_array)
    if use_scaling:
        if scaler is not None:
            selection_array = normalize_data(selection_array, scaler)
    print(selection_array.shape)
    cols = list()
    for i in range(n_in, 0, -1):
        cols.append(selection_array[-i, :])
    concat = np.hstack(cols)
    print(concat.shape)
    return concat

def add_indicators_to_data(selection_array):
    """
    Adds all indicators to the passed data array.
    This DELETES the first lines of data because NaN rows are dropped (30 for now, but specified by max(timeperiod)
    :param selection_array:
    :return:
    """
    selection_array = np.array(selection_array, dtype='f8')
    selection_array = add_SMA_indicator_to_data(selection_array, close_index=0, timeperiod=30)  # 1 column
    selection_array = add_BBANDS_indicator_to_data(selection_array, close_index=0)  # 3 columns
    selection_array = add_RSI_indicator_to_data(selection_array, close_index=0)  # 1 column
    selection_array = add_OBV_indicator_to_data(selection_array, close_index=0, volume_index=6)  # 1column
    selection_array = drop_NaN_rows(selection_array)
    return selection_array

def add_SMA_indicator_to_data(data, close_index=0, timeperiod=30):
    out = np.expand_dims(talib.SMA(data[:, close_index], timeperiod=timeperiod), axis=1)
    return np.concatenate([data, out], axis=1)


def add_RSI_indicator_to_data(data, close_index=0, timeperiod=14):
    out = np.expand_dims(talib.RSI(data[:, close_index], timeperiod=timeperiod), axis=1)
    return np.concatenate([data, out], axis=1)


def add_BBANDS_indicator_to_data(data, close_index=0, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0):
    upperband, middleband, lowerband = talib.BBANDS(data[:, close_index], timeperiod=timeperiod, nbdevup=nbdevup,
                                                    nbdevdn=nbdevdn, matype=matype)
    list = (
        data, np.expand_dims(upperband, axis=1), np.expand_dims(middleband, axis=1), np.expand_dims(lowerband, axis=1))
    return np.concatenate(list, axis=1)


def add_OBV_indicator_to_data(data, close_index=0, volume_index=6):  # Volume indicator
    out = np.expand_dims(talib.OBV(data[:, close_index], data[:, volume_index]), axis=1)
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
