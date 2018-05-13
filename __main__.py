import pandas as pd

import bot.parser
from bot_ai import neural
from util import data_enhancer
from util.printer import eprint

if __name__ == '__main__':
    eprint('start')
    data = pd.read_csv('data/.coinbaseUSD.csv')
    neur = neural.Neural()
    neur.build_binary_model(data, data_enhancer.make_binary_labels(data.loc[1]))