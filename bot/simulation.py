import logging
import pandas as pd
from collections import defaultdict

from bot import API_offline
from bot import API

from matplotlib import pyplot as plt

logger = logging.getLogger('simulation')

class Simulation:

    def __init__(self, dollar_balance, disable_fees=False, offline=False, label='bot_name'):
        self.label = label
        self.order_history = dict()
        self.account_standing_history = pd.DataFrame(columns=['date', 'dollars', 'account worth'])
        self.time_period = 300
        self.dollar_balance = dollar_balance
        self.disable_fees = disable_fees
        self.currency_balance = defaultdict(lambda:0)
        self.maker_fee = 0.001 #maker fee from https://poloniex.com/fees/ (lowest trade volume)
        self.taker_fee = 0.002 #taker fee from https://poloniex.com/fees/
        self.offline = offline

    def buy_amount(self, pair, amount_in_currency):
        """
        Buys at max the given amount for the specified pair in that currency simulated
        :param pair: the crypto pair
        :param amount_in_currency: the maximum amount to be bought, if this costs more than dollar is available spend all money
        :return: None
        """
        price_for_one_unit = self._price_for_one_unit(pair)
        if self.disable_fees:
            spending_actual = price_for_one_unit * amount_in_currency
        else:
            spending_actual = price_for_one_unit * amount_in_currency + price_for_one_unit * amount_in_currency * self.taker_fee #taker fee
        if self.dollar_balance < spending_actual:
            spending_actual = self.dollar_balance
        self.dollar_balance -= spending_actual

        if not self.disable_fees:
            spending_actual -= spending_actual*self.taker_fee #
        bought = spending_actual/price_for_one_unit
        currency = self.extract_second_currency_from_pair(pair)
        self.currency_balance[currency] += bought
        logger.info('{}: Bought {} of {} for {}'.format(self.label, bought, currency, spending_actual))
        logger.info('{}: You now have {} of {}'.format(self.label, self.currency_balance[currency], currency))

    def buy_with_amount(self, pair, amount_in_dollar):
        """
        buy crpyto specified by pair with amount of dollars (the other pair part)

        :param pair: the pair with the crpyto
        :param amount_in_dollar: max amount in dollars
        """
        if self.dollar_balance < amount_in_dollar:
            amount_in_dollar = self.dollar_balance
        price_for_one_unit = self._price_for_one_unit(pair)
        self.dollar_balance -= amount_in_dollar
        if not self.disable_fees:
            amount_in_dollar -= amount_in_dollar*self.taker_fee #use taker fee in this simulation
        bought = amount_in_dollar/price_for_one_unit
        currency = self.extract_second_currency_from_pair(pair)
        self.currency_balance[currency] += bought
        logger.info('{}: Bought {} for {}$'.format(self.label, currency, amount_in_dollar))
        logger.info('{}: You now have {} {}'.format(self.label, self.currency_balance[currency], currency))

    def _price_for_one_unit(self, pair):
        if self.offline:
            return API_offline.receive_latest_pair_price(pair, self.time_period)
        else:
            return  API.receive_latest_pair_price(pair, self.time_period)

    def sell(self, currency, amount):
        """
        Sell currency if there is any
        :param currency:
        :param amount:
        :return:
        """
        if currency in self.currency_balance:
            if self.currency_balance[currency] < amount:
                amount = self.currency_balance[currency]
            self.currency_balance[currency] -= amount
            price_for_one_unit = self._price_for_one_unit("USDT_"+currency)
            earning = price_for_one_unit * amount
            if not self.disable_fees:
                earning -= earning*self.taker_fee #use taker fee in this simulation
            self.dollar_balance += earning
            logger.info('{}: Sold {} {} for {}$'.format(self.label, amount, currency, earning))
            logger.info('{}: You now own {} of {}'.format(self.label, self.currency_balance[currency], currency))
        else:
            logger.error('{}: You do not possess {}'.format(self.label, currency))

    def withdraw(self, amount): #don't. its too pricey, 25$ for one withdrawal... puh
        self.dollar_balance -= amount
        if not self.disable_fees:
            self.dollar_balance -= 25.0 #fee from https://tether.to/fees/ for bank account withdrawal, and in theory only one per day


    def deposit_money(self, amount):
        """
        This resembles the trading token USDT
        :param amount: the amount of money to bank
        """
        self.dollar_balance += amount

    def get_currency_balance(self, currency):
        return self.currency_balance[currency]

    def get_dollar_balance(self):
        return self.dollar_balance

    def get_account_worth(self):
        """
        calculates the accounts worth by adding all currency standings multiplyed by their price
        :return: the worth in dollars (USDT tbh.)
        """
        sum = self.dollar_balance
        for cur in self.currency_balance:
            if cur == 'USDT':
                continue
            price_for_one_unit = self._price_for_one_unit("USDT_"+cur)
            sum += self.currency_balance[cur] * price_for_one_unit
        return sum

    def extract_first_currency_from_pair(self, pair):
        return pair.split('_')[0]

    def extract_second_currency_from_pair(self, pair):
        return pair.split('_')[1]

    def perform_action(self, date, action):
        """
        performs the given action within simulation for the given date and adds it in the order_history
        :param date: the date of action
        :param action: the action to take
        :return: None
        """
        pair, actionstr, amount, stop_loss = action
        if not pair in self.order_history:
            self.order_history[pair] = dict() # Make a dict if there are no orders for that pair yet
        if not date in self.order_history[pair]:
            self.order_history[pair][date] = action # add the action with date as key to the pair dictionary
        else:
            logger.warning('{}: Action already taken for date {}'.format(self.label, date))
            return
        if actionstr is 'hold':
            logger.info('{}: Holding {}'.format(self.label, pair))
        elif actionstr is 'buy':
            self.buy_amount(pair, amount)
        elif actionstr is 'sell':
            if amount > 0:
                cur = self.extract_second_currency_from_pair(pair)
                self.sell(cur, amount)

    def update_account_standing_history(self, date):
        if not self.account_standing_history['date'].isin([date]).any():
            self.account_standing_history.loc[len(self.account_standing_history)] = [date, self.dollar_balance, self.get_account_worth()]


    def print_trades(self):
        """
        Print all made trades in the most none pretty way
        Form follows function?
        """
        if self.order_history is not None and len(self.order_history) > 0:
            for pair, datedict in self.order_history.items():
                print('actions made for %s:' % pair)
                for date, action in datedict.items():
                    pair, actionstr, amount, stop_loss = action
                    print(date, actionstr, amount, 'stop-loss:', stop_loss)

    def save(self):
        pass

    def restore(self):
        pass

    def plot_account_history(self, title):
        """
        Visualizes the account standing history by ploting both amount of dollars and account worth
        :param title: the title of the chart
        """
        plt.plot(self.account_standing_history['date'], self.account_standing_history['dollars'], label='dollars')
        plt.plot(self.account_standing_history['date'], self.account_standing_history['account worth'], label='Account worth')
        plt.legend()
        plt.title(title)
        plt.show(block=False)


    def calc_win_margin_price(self, price, up_down_signum):
        """
        Calculate the minimum needed price to make a profit
        :param price: the price
        :param up_down_signum: if the trend goes up or down
        :return:
        """
        if self.disable_fees:
            return price
        if up_down_signum == 1: #action:buy => taker_fee
            margin_price = price + price*self.taker_fee
        else:
            margin_price = price - price*self.taker_fee
        return margin_price

