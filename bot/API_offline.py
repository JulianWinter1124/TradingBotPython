import logging
import time
from urllib.error import HTTPError, URLError

import pandas as pd

import directory
logger = logging.getLogger('API_offline')
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
    return lag

def download_and_save_data(abs_path, pair, time_period):
    logger.info('Downloading {} data...'.format(pair))
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
            logger.warning('File {} exists. If you want to get the latest data, remove it from data/'.format(abs_path))
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
        logger.error('NO NEW DATA AVAILABLE. Since this is offline mode you can shutdown the bot now')
    else:
        return df.loc[(df['date'] >= start_date) & (df['date'] <= end_date)].iloc[0:-lag]





