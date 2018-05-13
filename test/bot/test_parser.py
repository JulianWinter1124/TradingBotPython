import sys
import time
import unittest

from bot.parser import Parser
from util.printer import eprint


class ParserTest(unittest.TestCase):
    eprint("starting test")
    parser = Parser()
    parser.start()  # since parser is a thread on its on start this way. It's listening to input now
    time.sleep(5)
    print('settings timebank 10000\n\
settings time_per_move 100\n\
settings player_names player0\n\
settings your_bot player0\n\
settings candle_interval 1800\n\
settings candles_total 720\n\
settings candles_given 336\n\
settings initial_stack 1000\n\
settings candle_format pair,date,high,low,open,close,volume',file=sys.stdout)  # game settings taken from https://docs.riddles.io/crypto-trader/examples
    print('update game next_candles BTC_ETH,1516147200,0.095,0.09181,0.09219501,0.09199999,481.51276914;USDT_ETH,1516147200,1090.1676815,1022.16791604,1023.1,1029.99999994,1389783.7868468;USDT_BTC,1516147200,11600.12523891,11032.9211865,11041.42197477,11214.06052489,4123273.6568455',file=sys.stdout)
    print('update game next_candles BTC_ETH,1516149000,0.0930391,0.09,0.09199999,0.090488,343.77636407;USDT_ETH,1516149000,1044.14536025,977,1035.49999991,983.18751591,1337143.7718011;USDT_BTC,1516149000,11260.47834099,10837.42368984,11225.06052489,10847.30000032,3283745.1200703',file=sys.stdout)
    sys.stdout.write('asdas')