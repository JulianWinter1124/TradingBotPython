import multiprocessing
import math
import util.data_collector as dc


def mp_handler():
    data_collector = dc.DataCollector()
    p = multiprocessing.Process(target=data_collector.run, args=None) # arguments later?


if __name__ == '__main__':
    mp_handler()