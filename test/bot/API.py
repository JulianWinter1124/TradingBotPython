import unittest
import bot.API as API


class APITest(unittest.TestCase):

    def test_receive_latest_pair_price(self):
        print(API.receive_latest_pair_price('USDT_BTC', 300))
        print(API.receive_latest_pair_price('USDT_BTC', 900))
        print(API.receive_latest_pair_price('USDT_BTC', 1800))
        print(API.receive_latest_pair_price('USDT_BTC', 7200))


    def test_receive_currency_trading_info(self):
        print(API.receive_currency_trading_info('BTC'))
        print(API.receive_currency_trading_info('ETH'))
        print(API.receive_currency_trading_info('LTC'))

    def test_receive_currency_list(self):
        print(API.receive_currency_list())