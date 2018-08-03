import logging
from collections import defaultdict


class State:

    def __init__(self, balance):
        self.usd_balance = balance  # balance in US dollar
        self._currency_balance = defaultdict(lambda: 0)

    def deposit_currency(self, currency, amount):
        self._currency_balance[currency] += amount

    def withdraw_currency(self, currency, amount):
        if currency in self._currency_balance:
            if self._currency_balance[currency] >= amount:
                self._currency_balance -= amount
            else:
                logging.error('Cannot withdraw:' + amount + currency
                              + 'You only own: %d' % self._currency_balance[currency])
        else:
            logging.error('You dont own the currrency' + currency)

    def get_balance(self, currency):
        return self._currency_balance[currency]
