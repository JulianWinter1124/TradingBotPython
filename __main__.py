import pandas as pd

import bot.parser
from bot_ai import neural
from util import data_enhancer as de
from matplotlib import pyplot as plt
from util.printer import eprint
import numpy as np

if __name__ == '__main__':
    #data = pd.read_csv('data/.coinbaseUSD.csv', header=None)
    #temp = data.loc[0:1000]
    #temp.to_csv('data/cbUSDtest.csv', index=False, header=None)
    dataframe = pd.read_csv('data/cbUSDtest.csv', header=None, names=['date', 'price', 'volume'])
    data = dataframe.values
    train, test = de.split_dataset_in_training_and_test(data, 0.67)

    train_X, train_y = de.create_shifted_datasets(train)
    test_X, test_y = de.create_shifted_datasets(test)

    prices = data[:, 1]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

    neur = neural.Neural()
    model = neur.build_model(train_X)
    history = neur.train_model(model, train_X, train_y, test_X, test_y)

    plt.plot(history.history['loss'], label='train') #OmegaLUL
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()

    lul = model.predict(test_X)
    print(lul)


