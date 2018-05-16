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


def create_shifted_datasets(dataset, shift=1, prediction_index=1):
    size = len(dataset)
    print(dataset.shape)
    Y = dataset[shift:size, prediction_index]
    X = dataset[0:size - shift, :]
    return X, Y


def split_dataset_in_training_and_test(dataset, train_size: float = 0.67):
    data_index = int(len(dataset) * train_size)
    return dataset[0:data_index, :], dataset[data_index:len(dataset), :]
