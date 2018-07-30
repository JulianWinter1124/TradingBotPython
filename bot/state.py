import logging
import time
from urllib.error import HTTPError, URLError

import pandas as pd


class State:

    def __init__(self, balance):
        self.usd_balance = balance  # balance in US dollar

    def buy(self, pair, amount):

