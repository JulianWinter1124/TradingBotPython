import logging
from collections import defaultdict

import numpy as np

from bot import API


class Simulation:

    def __init__(self, dollar_balance, disable_fees=False):
        self.time_period = 300
        self.dollar_balance = dollar_balance
        self.disable_fees = disable_fees
        self.currency_balance = defaultdict(lambda:0)
        self.maker_fee = 0.001
        self.taker_fee = 0.002

    def buy_amount(self, pair, amount_in_currency):
        price_for_one_unit = API.receive_latest_pair_price(pair, self.time_period)
        if self.disable_fees:
            spending_actual = price_for_one_unit * amount_in_currency
        else:
            spending_after_fee = price_for_one_unit * amount_in_currency
            spending_actual = spending_after_fee / (1-self.taker_fee) #taker fee
        if self.dollar_balance < spending_actual:
            spending_actual = self.dollar_balance
        self.dollar_balance -= spending_actual
        if not self.disable_fees:
            spending_actual -= spending_actual*self.taker_fee #taker fee from https://poloniex.com/fees/
        bought = spending_actual/price_for_one_unit
        currency = self.extract_buying_currency_from_pair(pair)
        self.currency_balance[currency] += bought
        logging.info('Bought' + currency + 'for%d' %spending_actual)
        logging.info('You now have: %d' %self.currency_balance[currency])

    def buy_with_amount(self, pair, amount_in_dollar):
        if self.dollar_balance < amount_in_dollar:
            amount_in_dollar = self.dollar_balance
        price_for_one_unit = API.receive_latest_pair_price(pair, self.time_period)
        self.dollar_balance -= amount_in_dollar
        if not self.disable_fees:
            amount_in_dollar -= amount_in_dollar*self.taker_fee #taker fee from https://poloniex.com/fees/
        bought = amount_in_dollar/price_for_one_unit
        currency = self.extract_buying_currency_from_pair(pair)
        self.currency_balance[currency] += bought
        logging.info('Bought' + currency + 'for%d' %amount_in_dollar)
        logging.info('You now have: %d' %self.currency_balance[currency])


    def sell(self, currency, amount):
        if currency in self.currency_balance:
            if self.currency_balance[currency] < amount:
                amount = self.currency_balance[currency]
            self.currency_balance[currency] -= amount
            if not self.disable_fees:
                amount -= amount*self.maker_fee #maker fee from https://poloniex.com/fees/ (lowest trade volume)
            price_for_one_unit = API.receive_latest_pair_price('USDT_'+currency, self.time_period)
            earning = price_for_one_unit * amount
            self.dollar_balance += earning
        else:
            logging.error('You do not possess ' + currency)

    def withdraw(self, amount): #don't. its too pricey, 25$ for one withdrawal... puh
        self.dollar_balance -= amount


    def deposit_money(self, amount):
        """
        This resembles the trading token
        :param amount: the amount of money to bank
        """
        self.dollar_balance += amount

    def get_currency_balance(self, currency):
        return self.currency_balance[currency]

    def get_dollar_balance(self):
        return self.dollar_balance

    def get_account_worth(self):
        sum = self.dollar_balance
        for cur in self.currency_balance:
            price_for_one_unit = API.receive_latest_pair_price('USDT_' + cur, self.time_period)
            sum += self.currency_balance[cur] * price_for_one_unit
        return sum

    def extract_buying_currency_from_pair(self, pair):
        return pair.split('_')[1]

    def extract_currency_to_buy_from_pair(self, pair):
        return pair.split('_')[0]

    def perform_action(self, action):
        pair, actionstr, amount, tanhrisk = action
        if actionstr is 'buy':
            self.buy_with_amount(pair, amount)
            #TODO continue


#End of class

def calc_win_margin_price(price, up_down_signum, taker_fee=0.002, maker_fee=0.001):
    if up_down_signum == 1: #action:buy => taker_fee
        margin_price = np.float64(np.float64(price) / (1-np.float64(taker_fee)))
    else:
        margin_price = np.float64(np.float64(price) * (1-np.float64(maker_fee)))
    return margin_price