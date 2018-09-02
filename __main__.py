import getopt
import sys
import time

from bot.simulation import Simulation
from bot.tradingbot import TradingBot
from util.config_manager import BotConfigManager


def main():
    config = BotConfigManager()
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hr", ["reconfig"])
    except getopt.GetoptError:
        print('unknwon options.')
        print('run "__main__.py -h" for help')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('usage: python __main__.py [option]')
            print('Options:')
            print('--reconfig\t reconfigures the bot by loading vars from config_manager.init_vars.')
            sys.exit()
        elif opt in ("-r", "--reconfig"):
            config.init_variables()
            config.save_config()

    success = config.load_config()
    if not success:
        print('no existing config found. Edit params in config_manager.py and restart bot with args --reconfig')
        raise SystemExit
    else:
        print('loaded config successfully. run again with --reconfig if needed')

    minimum_loop_time = config.timesteps/1.0 #Find better value?
    tradingbot = TradingBot(config)
    simulation = Simulation(500, False)
    try:
        while True:
            exec_time = tradingbot.run(simulation)
            print('Loop execution took:', exec_time,
                  'seconds. Waiting %f seconds. (It is safe to force shutdown now)' % max(minimum_loop_time - exec_time,
                                                                                          0))
            time.sleep(max(minimum_loop_time - exec_time, 0))

    except KeyboardInterrupt:
        print('Keyboard interrupt, shutting down')
        tradingbot.perform_shutdown()

if __name__ == '__main__':
    main()