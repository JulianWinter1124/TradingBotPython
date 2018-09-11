import logging
import pandas as pd
from matplotlib import pyplot as plt
from collections import defaultdict

from bot import API


class Simulation:

    def __init__(self, dollar_balance, disable_fees=False):
        self.order_history = dict()
        self.account_standing_histoy = dict()
        self.time_period = 300
        self.dollar_balance = dollar_balance
        self.disable_fees = disable_fees
        self.currency_balance = defaultdict(lambda:0)
        self.maker_fee = 0.001 #maker fee from https://poloniex.com/fees/ (lowest trade volume)
        self.taker_fee = 0.002 #taker fee from https://poloniex.com/fees/

    def buy_amount(self, pair, amount_in_currency):
        """
        Buys at max the given amount for the specified pair in that currency simulated
        :param pair: the crypto pair
        :param amount_in_currency: the maximum amount to be bought, if this costs more than dollar is available spend all money
        :return: None
        """
        price_for_one_unit = API.receive_latest_pair_price(pair, self.time_period)
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
        logging.info('Bought' + currency + 'for%d' %spending_actual)
        logging.info('You now have: %d' %self.currency_balance[currency])

    def buy_with_amount(self, pair, amount_in_dollar):
        """
        buy crpyto specified by pair with amount of dollars (the other pair part)

        :param pair: the pair with the crpyto
        :param amount_in_dollar: max amount in dollars
        """
        if self.dollar_balance < amount_in_dollar:
            amount_in_dollar = self.dollar_balance
        price_for_one_unit = API.receive_latest_pair_price(pair, self.time_period)
        self.dollar_balance -= amount_in_dollar
        if not self.disable_fees:
            amount_in_dollar -= amount_in_dollar*self.taker_fee #use taker fee in this simulation
        bought = amount_in_dollar/price_for_one_unit
        currency = self.extract_second_currency_from_pair(pair)
        self.currency_balance[currency] += bought
        logging.info('Bought' + currency + 'for%d' %amount_in_dollar)
        logging.info('You now have: %d' %self.currency_balance[currency])


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
            if not self.disable_fees:
                amount -= amount*self.taker_fee #use taker fee in this simulation
            price_for_one_unit = API.receive_latest_pair_price('USDT_'+currency, self.time_period)
            earning = price_for_one_unit * amount
            self.dollar_balance += earning
        else:
            logging.error('You do not possess ' + currency)

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
            price_for_one_unit = API.receive_latest_pair_price('USDT_' + cur, self.time_period)
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
            self.account_standing_histoy[pair] = dict()
        if not date in self.order_history[pair]:
            self.order_history[pair][date] = action # add the action with date as key to the pair dictionary
            self.account_standing_histoy[pair][date] = pd.DataFrame(columns=['dollars', 'account worth'])
        else:
            print('Action already taken.')
            return

        if actionstr is 'hold':
            print('Not trading anything', pair)
        elif actionstr is 'buy':
            self.buy_amount(pair, amount)
        elif actionstr is 'sell':
            if amount > 0:
                cur = self.extract_second_currency_from_pair(pair)
                self.sell(cur, amount)
        print("dollar balance is now:", self.dollar_balance)
        worth = self.get_account_worth()
        print('account worth is now:', worth)
        self.account_standing_histoy[pair][date].append([[self.dollar_balance, worth]])
        print(self.account_standing_histoy[pair][date])

    def print_trades(self):
        """
        Print all made trades in the most none pretty way
        Form follows function?
        :return:
        """
        if self.order_history is not None and len(self.order_history) > 0:
            for pair, datedict in self.order_history.items():
                print('actions made for %s:' % pair)
                for date, action in datedict.items():
                    pair, actionstr, amount, stop_loss = action
                    print(date, actionstr, amount, 'stop-loss:', stop_loss)

    def plot_account_history(self):
        for pair in self.account_standing_histoy.keys():
            for date in self.account_standing_histoy[pair].keys():
                plt.plot(x=date, y=self.account_standing_histoy[pair][date]['dollars'], color='red')
                plt.plot(x=date, y=self.account_standing_histoy[pair][date]['account worth'], color='blue')
            plt.show()

    def save(self):
        pass

    def restore(self):
        pass

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

