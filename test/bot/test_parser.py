import sys
import time
import unittest

from bot.parser import Parser
from util.printer import eprint


class CollectorTest(unittest.TestCase):

    def test_lul(self):
        from util.data_collector_v2 import DataCollector
        coll = DataCollector('data', ['USDT_BTC', 'USDT_ETH'], [1405699200, 1405699200], [9999999999, 9999999999],
                             time_periods=[300, 300], overwrite=False)
        from util.data_processor import DataProcessor
        proc = DataProcessor(database_filepath=coll.filepath, output_filepath='data/finished_data.h5')
        from multiprocessing import Process
        p1 = Process(target=coll.run_unmodified_loop)
        p2 = Process(target=proc.run)
        p1.start()
        p2.start()
        p2.join()