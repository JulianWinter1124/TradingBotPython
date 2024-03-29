import logging
import time
from urllib.error import HTTPError, URLError

import pandas as pd

BASE_URL = 'https://poloniex.com/public?command=returnChartData'

#This class only uses the poloniex public API, so no secret or api key is required.
#For completion reasons I included the poloniex_API.py (WHICH IS NOT MINE)


def receive_latest_pair_price(pair, time_period):
    """
    receives only the latest close price for the specified currency pair
    Note: a Pandas error might be due to an unknown pair. i don't want to catch this as this is unintended behavior
    :param pair: the crpyto pair to receive data from.
    :param time_period: valid: 300, 900, 1800, 7200, 14400, and 86400
    :return: the latest available price for the given pair
    """
    current = int(time.time())

    count = 0
    while(True):
        try:
            url = build_url(pair, current-time_period*5, current, time_period) #5 is making sure new data is available
            df = pd.read_json(url, convert_dates=False)
        except (HTTPError, URLError) as e:
            logging.error('error retrieving latest price. Trying again in %d seconds.' % (count))
            time.sleep(count)
            count += 1
            continue
        return df['close'].tail(1).values[0]


def receive_pair_data(pair, start_date, end_date, time_period):
    """
    Receives all pair data within the specified dates.
    Note: a Pandas error might be due to an unknown pair
    :param pair: the currency pair. e.g. USDT_BTC
    :param start_date: start date in unix format
    :param end_date: end date in unix format (enter something high)
    :param time_period:time period in which data comes in. valid=300, 900, 1800, 7200, 14400, and 86400
    :return: A dataframe containing all data for the currency pair
    """
    url = build_url(pair, start_date, end_date, time_period)
    count = 0
    while(True):
        try:
            df = pd.read_json(url, convert_dates=False)  # TODO: catch errors
        except (HTTPError, URLError) as e:
            logging.error('error retrieving latest price for pair' + pair + 'Trying again in %d seconds.' %(count))
            time.sleep(count)
            count+=1
            continue
        return df

def build_url(pair, start_date, end_date, time_period) -> str:
    """
    Builds the return_chart_data url for poloniex
    see: https://poloniex.com/support/api/
    :param pair: the currency pair. e.g. USDT_BTC
    :param start_date: start date in unix format
    :param end_date: end date in unix format (enter something high)
    :param time_period:time period in which data comes in. valid=300, 900, 1800, 7200, 14400, and 86400
    :return:
    """
    return BASE_URL + '&currencyPair=' + str(pair) + '&start=' + str(start_date) + '&end=' + str(
        end_date) + '&period=' + str(time_period)

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




