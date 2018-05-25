import pandas as pd
import numpy as np


def make_binary_labels(closing_price_column):
    """Calculate labels (-1 for down or 1 for up)
    :param closing_price_column: The data the labels are generated from
    :return: the labels in a list with length
    """
    label_list = [np.sign(closing_price_column[i] - closing_price_column[i - 1]) for i in
                  range(1, len(closing_price_column))]
    return label_list


def make_ranged_labels(closing_price_column):
    """
    :param closing_price_column: The data the labels are generated from
    :return: difference between prices unified to (-1, 1) in R
    """
    high = np.max(closing_price_column)
    low = np.min(closing_price_column)  # Bei groesser werdenden Daten evtl. aendern
    label_list = [(closing_price_column[i] - closing_price_column[i - 1]) / (high - low) for i in
                  range(1, len(closing_price_column))]
    return label_list


def create_shifted_datasets(dataset, shift=1):
    print(dataset.shape)
    Y = dataset[shift:, :]
    X = dataset[0:-shift, :]
    return X, Y


def split_dataset_in_training_and_test(dataset, train_size: float = 0.80):
    data_index = int(len(dataset) * train_size)
    return dataset[0:data_index, :], dataset[data_index:len(dataset), :]


def series_to_supervised(df: pd.DataFrame, n_in=1, n_out=1, dropnan=True):
    """
    Alterd from:
    Frame a time series as a supervised learning dataset.
    Arguments:
        data: Sequence of observations as a DataFrame
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """
    var_names = df.columns
    cols, names = list(), list()
    for i in range(n_in, -n_out-1, -1):
        column = df.shift(i)
        # column['date'] = column['date'].fillna(0).astype(int)
        cols.append(column)
        names += [s + '(%+d)' % (-i) for s in var_names]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg
