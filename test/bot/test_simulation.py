import unittest

from bot import simulation
from bot.simulation import Simulation


class TestSimulation(unittest.TestCase):

    def create(self):
        self.simulation = Simulation(0, False)
        self.simulation.deposit_money(1000.9)

    def test_deposit_money(self):
        self.create()
        self.assertTrue(self.simulation.get_dollar_balance() == 1000.9)

    def test_buy_with_amount(self):
        self.create()
        self.simulation.buy_with_amount('USDT_BTC', 100)
        self.assertTrue(self.simulation.get_dollar_balance() == 1000.9 - 100)

    def test_buy_amount(self):
        self.create()
        self.simulation.buy_amount('USDT_BTC', 0.02)
        print(self.simulation.get_dollar_balance())
        self.assertEqual(0.02, self.simulation.get_currency_balance('BTC'))

    def test_sell(self):
        self.create()
        print(self.simulation.get_dollar_balance())
        self.simulation.buy_amount('USDT_BTC', 0.03)
        self.simulation.sell('BTC', 0.03)
        self.assertEqual(0.0, self.simulation.get_currency_balance('BTC'))
        print(self.simulation.get_dollar_balance())

    def test_get_account_worth(self):
        self.create()
        print(self.simulation.get_dollar_balance())
        self.simulation.buy_amount('USDT_BTC', 0.03)
        print(self.simulation.get_dollar_balance())
        print(self.simulation.get_account_worth())

class TestSimulationMethods(unittest.TestCase):

    def test_calc_win_margin_price(self):
        print(simulation.calc_win_margin_price(1, +1))
        print(simulation.calc_win_margin_price(1, -1))
        print(simulation.calc_win_margin_price(1, 0))
