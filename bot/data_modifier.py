import logging

import numpy as np
import talib
from sklearn.preprocessing import MinMaxScaler, StandardScaler
logger = logging.getLogger('data_modifier')

def normalize_data_MinMax(data):
    """
    Normalizes the given data with a minMax scaler
    :param data: the data to scale
    :return: scaled data, used scaler
    """
    scaler = MinMaxScaler()
    logger.info('scaler fit on data with shape: {}'.format(data.shape))
    data = scaler.fit_transform(data)
    return data, scaler


def normalize_data_Standard(data):
    """
    Normalizes the given data with a Standard deviation scaler
    :param data: the data to scale
    :return: scaled data, used scaler
    """
    scaler = StandardScaler()
    logger.info('scaler fit on data with shape: {}'.format(data.shape))
    data = scaler.fit_transform(data)
    return data, scaler


def normalize_data(data, scaler):
    """
    Scale the data with the given scaler
    :param data: the data in the right shape
    :param scaler: the scaler used on same shaped data
    :return: scaled data
    """
    if (scaler != None):
        return scaler.transform(data)
    else:
        return data


def reverse_normalize_prediction(prediction, label_index_in_original_data, n_features, scaler):
    """
    Reverses scaling of labels by bringing labels in the right shape one by one and scales them. at last put all scaled labels together
    :param prediction: the prediction data
    :param label_index_in_original_data: the label index in the original data
    :param n_features: the number of columns in the original data
    :param scaler: the scaler used to transform the data
    :return: the unscaled predictions in the same shape as prediction
    """
    cols = list()
    for i in range(prediction.shape[1]):
        pred_column = prediction[:, i]
        dummy_data = np.zeros(shape=(prediction.shape[0], n_features)) #Make array with same shape as original data
        dummy_data[:, label_index_in_original_data] = pred_column
        scaled_dummy_data = scaler.inverse_transform(dummy_data) #reverse scale of dummy array containing the one column of predictions(=labels in original data)
        unscaled_label_column = np.atleast_2d(scaled_dummy_data[:, label_index_in_original_data]).T #Dont
        cols.append(unscaled_label_column) # append inverse scaled rpedictions to result list
    concat = np.concatenate(cols, axis=1) #concatenate list
    return concat


def data_to_supervised_timeseries(data, n_in=1, n_out=1, n_out_jumps=1, label_columns_index=0):
    """
    Converts the given data to a supervised timeseries. Inspired from https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
    :param data: the data to convert in a numpy array
    :param n_in: the number of timesteps you look back in the past. (1 or higher)
    :param n_out: the number of timesteps+1 you look in the future. Allowed are -1 (no labels) or higher
    :param n_out_jumps: the jumps you make in the future (e.g. 2 means t+1, t+3, t+5.... are predicted)
    :param label_columns_index: the index where the label is stored in the data
    :return: the finished timeseries numpy array
    """
    cols = list()
    for i in range(n_in, -n_out - 1, -1): #Work from left to right
        if i <= 0:
            column = np.roll(data, i*n_out_jumps, axis=0) #'Roll' the data to create the timeseries
            column = column[:, [label_columns_index]]  # The future points may only contain labels
        else:
            column = np.roll(data, i, axis=0)
        cols.append(column) #Append the columns to the list
    concat = np.concatenate(cols, axis=1) #Concat all columns
    if n_out == 0 or n_out == -1:
        concat = concat[n_in:]# numbers that would be NaN are always dropped
    else:
        concat = concat[n_in:-n_out*n_out_jumps] #Drop
    concat = np.atleast_2d(concat)
    logger.info('Converted data to timeseries. It has the shape {}'.format(concat.shape))
    return concat

def data_to_single_column_timeseries_without_labels(data, n_in, scaler, label_transform_function=None, use_scaling=True, use_indicators=True, label_index=0):
    """
    Creates a single row of timeseries data. this can be used to predict the latest date.
    :param data: all data, or at least the latest 30+n_in rows (in case indicators are used)
    :param n_in: the number of steps you look back in the past
    :param scaler: the scalers that have been used by the data_processor class
    :param drop_columns_indices:
    :param use_scaling:
    :param use_indicators:
    :return: the timeseries row without labels
    """
    selection_array = data[-(n_in+30):, :]
    if use_indicators:  # adding indicators
        selection_array = add_indicators_to_data(selection_array)
    if label_transform_function is not None:
        selection_array = add_custom_label_to_data(selection_array, label_transform_function, label_index)
    if use_scaling: #scaling data
        if scaler is not None:
            selection_array = normalize_data(selection_array, scaler)
    cols = list()
    for i in range(n_in, 0, -1):
        cols.append(selection_array[-i, :])
    concat = np.hstack(cols)
    return np.atleast_2d(concat)

def data_to_timeseries_without_labels(data, n_in, scaler, label_transform_function=None, use_scaling=True, use_indicators=True, label_index=0):
    """
    This scales the data, adds indicators, the custom label and also transforms it into a timeseries without labels. This is useful to test
    predictions on the whole dataset.
    :param data: all data
    :param n_in:
    :param scaler:
    :param label_transform_function:
    :param use_scaling:
    :param use_indicators:
    :param label_index:
    :return: the timeseries without labels
    """
    selection_array = data[:, :]
    if use_indicators:  # adding indicators
        selection_array = add_indicators_to_data(selection_array)
    if label_transform_function is not None:
        selection_array = add_custom_label_to_data(selection_array, label_transform_function, label_index)
    if use_scaling:
        if scaler is not None:
            selection_array = normalize_data(selection_array, scaler)
    cols = list()
    for i in range(n_in-1, -1, -1): #Work from left to right
        column = np.roll(selection_array, i, axis=0)
        cols.append(column) #Append the columns to the list
    concat = np.concatenate(cols, axis=1) #Concat all columns
    if n_in != 1:
        concat = concat[n_in-1:]# numbers that would be NaN are always dropped
    concat = np.atleast_2d(concat)
    return np.atleast_2d(concat)

def add_custom_label_to_data(data, label_transform_function, label_index): #Right now this is more like an indicator
    """
    Adds your custom label_transformation_function column to the given data array
    :param data:
    :param label_transform_function:
    :param label_index:
    :return:
    """
    selection_array = np.copy(data)
    new_label_column = label_transform_function(selection_array, label_index)
    selection_array = np.concatenate([selection_array, new_label_column], axis=1)
    #selection_array[:,[label_index,-1]] = selection_array[:,[-1,label_index]] #Swap label column so transformed label gets predicted
    return drop_NaN_rows(selection_array)


def add_indicators_to_data(selection_array):
    """
    Adds all indicators to the passed data array.
    This deletes the first lines of data because NaN rows are dropped (30 for now, specified by the maximum timeperiod in indicators
    :param selection_array: The selected array where indiccators are added
    :return: the original array with the added indicators at the right
    """
    selection_array = np.array(selection_array, dtype='f8') #Ta-Lib does not like other floats?
    selection_array = add_SMA_indicator_to_data(selection_array, close_index=0, timeperiod=30)  # 1 column
    selection_array = add_BBANDS_indicator_to_data(selection_array, close_index=0)  # 3 columns
    selection_array = add_RSI_indicator_to_data(selection_array, close_index=0)  # 1 column
    selection_array = add_OBV_indicator_to_data(selection_array, close_index=0, volume_index=6)  # 1column
    selection_array = add_LINEARREG_indicator_to_data(selection_array, close_index=0, timeperiod=14) # 1column
    selection_array = drop_NaN_rows(selection_array)
    return selection_array

# See https://mrjbq7.github.io/ta-lib/funcs.html for further details on every functions

def add_SMA_indicator_to_data(data, close_index=0, timeperiod=30): #Trend
    out = np.expand_dims(talib.SMA(data[:, close_index], timeperiod=timeperiod), axis=1)
    return np.concatenate([data, out], axis=1)


def add_RSI_indicator_to_data(data, close_index=0, timeperiod=14): #Momentum indicator
    out = np.expand_dims(talib.RSI(data[:, close_index], timeperiod=timeperiod), axis=1)
    return np.concatenate([data, out], axis=1)


def add_BBANDS_indicator_to_data(data, close_index=0, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0): #Trend
    upperband, middleband, lowerband = talib.BBANDS(data[:, close_index], timeperiod=timeperiod, nbdevup=nbdevup,
                                                    nbdevdn=nbdevdn, matype=matype)
    list = (
        data, np.expand_dims(upperband, axis=1), np.expand_dims(middleband, axis=1), np.expand_dims(lowerband, axis=1))
    return np.concatenate(list, axis=1)


def add_OBV_indicator_to_data(data, close_index=0, volume_index=6):  # Volume indicator
    out = np.expand_dims(talib.OBV(data[:, close_index], data[:, volume_index]), axis=1)
    return np.concatenate((data, out), axis=1)

def add_LINEARREG_indicator_to_data(data, close_index=0, timeperiod=14):  # Statistics indicator
    out = np.expand_dims(talib.LINEARREG(data[:, close_index], timeperiod), axis=1)
    return np.concatenate((data, out), axis=1)


def drop_NaN_rows(data): #drops als rows that contain NaN floats
    return data[~np.isnan(data).any(axis=1)]


def create_binary_labels(closing_price_column): #This is not used anymore
    """Calculate labels (-1 for down or 1 for up)
    :param closing_price_column: The data the labels are generated from
    :return: the labels in a list with length
    """
    label_list = [np.sign(closing_price_column[i] - closing_price_column[i - 1]) for i in
                  range(1, len(closing_price_column))]
    return label_list


def create_ranged_labels(closing_price_column): #Not used either
    """
    :param closing_price_column: The data the labels are generated from
    :return: difference between prices unified to (-1, 1) in R
    """
    high = np.max(closing_price_column)
    low = np.min(closing_price_column)  # Bei groesser werdenden Daten evtl. aendern
    label_list = [(closing_price_column[i] - closing_price_column[i - 1]) / (high - low) for i in
                  range(1, len(closing_price_column))]
    return label_list

def difference_label_transformation_function(data, label_column_index):
    """
    this functions calculates the difference between t and t+1
    :param data: the whole data
    :param label_column_index: the index at which the label is (close-index)
    :return: A np.Array with a single column containing the differences.
    """
    future_data = np.roll(data, -1, axis=0)
    transformed = np.subtract(future_data, data)[:, label_column_index]
    transformed[-1] = np.nan #replace with nan since this value is wrong
    return np.expand_dims(transformed, axis=1)
