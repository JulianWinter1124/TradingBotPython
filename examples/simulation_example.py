import time

from bot.simulation import Simulation
from bot.tradingbot import TradingBot
from util.config_manager import BotConfigManager

#this will be an example simulation with

if __name__ == '__main__':
    config = BotConfigManager()
    success = config.load_config()
    if not success:
        config.create_empty_config()
        print('Created empty config file, fill in your values')
        raise SystemExit
    minimum_loop_time = 300
    tradingbot = TradingBot(config)
    state = 'RUN'
    simulation = Simulation(100, False)
    try:
        while True:
            exec_time = tradingbot.run(state)
            print('Loop execution took:', exec_time, 'seconds. Waiting %f seconds.' % max(minimum_loop_time-exec_time, 0))
            time.sleep(max(minimum_loop_time-exec_time, 0))

    except KeyboardInterrupt:
        print('Keyboard interrupt, shutting down')
        tradingbot.perform_shutdown()