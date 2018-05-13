import datetime
import sys
import pandas as pd
import threading

from bot.move import Move
from util.printer import eprint


class Parser(threading.Thread):

    def __init__(self):
        threading.Thread.__init__(self)  # it's a thread!
        self.timebank = -1
        self.current_date: datetime
        self.stacks = pd.DataFrame(columns=['symbol', 'amount'])
        self.MAX_TIMEBANK = -1
        self.TIME_PER_MOVE = -1
        self.CANDLE_INTERVAL = -1
        self.CANDLE_FORMAT = []
        self.CANDLES_TOTAL = -1
        self.CANDLES_GIVEN = -1
        self.INITIAL_STACK = -1
        self.TRANSACTION_FEE = -1.0
        self.stacks = dict({})
        self.chart_data: pd.DataFrame = pd.DataFrame()

    def run(self):
        for line in sys.stdin:
            if len(line) == 0:
                continue
            eprint(line)  # test output
            parts = line.split(' ')
            if parts[0] == 'settings':
                self.parse_settings(parts[1], parts[2])
            elif parts[0] == 'update':
                if parts[1] == 'game':
                    self.parse_game_data(parts[2], parts[3])
            elif parts[0] == 'action':
                self.timebank = int(parts[2])
                move: Move = Move()
                print(str(move))
            else:
                eprint('Unknown command')

    def parse_settings(self, key: str, value: str):
        if key == 'timebank':
            time = int(value)
            self.MAX_TIMEBANK = time
            self.timebank = time
        elif key == 'time_per_move':
            self.TIME_PER_MOVE = int(value)
        elif key == 'candle_interval':
            self.CANDLE_INTERVAL = int(value)
        elif key == 'candle_format':
            self.CANDLE_FORMAT = value.split(',')
            self.chart_data = pd.DataFrame(columns=self.CANDLE_FORMAT)
            self.chart_data.set_index('pair')
        elif key == 'candles_total':
            self.CANDLES_TOTAL = int(value)
        elif key == 'candles_given':
            self.CANDLES_GIVEN = int(value)
        elif key == 'initial_stack':
            self.INITIAL_STACK = int(value)
        elif key == 'transaction_fee_percent':
            self.TRANSACTION_FEE = float(value)
        else:
            eprint('Could not parse: ', key, value)

    def parse_game_data(self, key: str, value: str):
        if key == 'next_candles':
            for chart_str in value.split(';'):
                self.update_chart(chart_str)
            eprint(self.chart_data)
        elif key == 'stacks':
            for stack in value.split(','):
                stack_arr = stack.strip().split(':')
                self.update_stacks(stack_arr[0], float(stack_arr[1]))
        else:
            eprint("Could not parse game data.")

    def update_stacks(self, symbol: str, amount: float):
        self.stacks[symbol] = amount

    def update_chart(self, candle: str):
        new_candle = pd.DataFrame([candle.split(',')], columns=self.CANDLE_FORMAT)
        eprint('the new row:\n', new_candle, '\n row ends.')
        self.chart_data = self.chart_data.append(new_candle, ignore_index=True, )