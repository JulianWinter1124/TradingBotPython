import logging
import time
from urllib.error import HTTPError, URLError

import pandas as pd

import directory

data = dict()
lag = 0

#This class only uses the poloniex public API, so no secret or api key is required.
#For completion reasons I included the poloniex_API.py (WHICH IS NOT MINE)

def init_global_lag(n_lag):
    global lag
    lag = n_lag

def decrease_global_lag():
    global lag
    lag -= 1

def download_and_save_data(abs_path, pair, time_period):
    print('Downloading data...')
    url = 'https://poloniex.com/public?command=returnChartData' + '&currencyPair=' + str(pair) + '&start=' + str(0) + '&end=' + str(
        9999999999) + '&period=' + str(time_period)
    dataframe = pd.read_json(url, convert_dates=False)
    directory.ensure_directory(abs_path)
    dataframe.to_csv(abs_path, index=False)

def load_data_and_download_if_not_existent(pair_list, time_period):
    for pair in pair_list:
        abs_path = directory.get_absolute_path('data/'+pair+'_'+str(time_period)+'.csv')
        if not directory.file_exists(abs_path):
            download_and_save_data(abs_path, pair, time_period)
        else:
            print('file exists')
        global data
        data[pair] = pd.read_csv(abs_path)


def receive_latest_pair_price(pair, time_period): #Please call load_data_and_download_if_not_existent first somewhere
    """
    Read the latest - lag close entry
    :param pair:
    :param time_period: unused, just so code has not to be changed
    :return:
    """
    return data[pair]['close'].iloc[-(1+lag)]


def receive_pair_data(pair, start_date, end_date, time_period):
    """

    :param pair:
    :param start_date:
    :param end_date:
    :param time_period:
    :return:
    """
    df = data[pair]
    if lag == 0:
        return df.loc[(df['date'] >= start_date) & (df['date'] <= end_date)]
    elif lag < 0:
        print('NO NEW DATA AVAILABLE')
    else:
        return df.loc[(df['date'] >= start_date) & (df['date'] <= end_date)].iloc[0:-lag]


def receive_currency_trading_info(currency): #Not really used right now
    """
    Receives trading info about a specific currency. Example format:
    delisted                   0
    depositAddress          None
    disabled                   0
    frozen                     0
    id                        28
    minConf                    1
    name                 Bitcoin
    txFee             0.00050000
    :param currency: a currency shortcut as returned in receive_currency_list()
    :return: A dataframe with trading info for the specified currency
    """
    while(True):
        try:
            currencies = pd.read_json('https://poloniex.com/public?command=returnCurrencies')
        except HTTPError:
            logging.error('error retrieving [' + currency + '] trading info. Trying again in %d seconds.' % 1)
            time.sleep(1)
            continue
        return currencies[currency]

def receive_currency_list(): #not used right now
    """
    Receeives a list of all currencies on poloniex
    see: https://poloniex.com/support/api/
    :return: all currencies as list
    """
    while (True):
        try:
            currencies = pd.read_json('https://poloniex.com/public?command=returnCurrencies')
        except HTTPError:
            logging.error('error retrieving currency list. Trying again in %d seconds.' % 1)
            time.sleep(1)
            continue
        return currencies.columns.values




