import getopt
import logging
import signal
import sys
import time

from bot import API_offline
from bot.simulation import Simulation
from bot.tradingbot import TradingBot
from util.config_manager import BotConfigManager


def main():
    signal.signal(signal.SIGINT, signal.default_int_handler)
    logging.getLogger().setLevel(logging.INFO)
    #logging.disable(logging.INFO) #If you want to disable INFO logging uncomment this
    logging.disable(logging.NOTSET) #if you want to display INFO again uncomment this
    config = BotConfigManager()
    offline = False
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hr", ["help", "reconfig", "offline, log"])
    except getopt.GetoptError:
        print('unknwon options.')
        print('run "__main__.py -h" for help')
        sys.exit(2)
    for opt, arg in opts:
        print(opt, arg)
        if opt in ('-h', '--help'):
            print('usage: python __main__.py [option]')
            print('Options:')
            print('--reconfig\t reconfigures the bot by loading vars from config_manager.init_vars.')
            sys.exit()
        elif opt in ("-r", "--reconfig"):
            config.init_variables()
            config.save_config()
        elif opt in ('--offline'):
            print('running in offline mode...')
            offline = True
            API_offline.init_global_lag(1000)
        elif opt in ('--log'):
            print("Setting log level to", arg)
            logging.getLogger().setLevel(arg)

    success = config.load_config()
    if not success:
        print('no existing config found. Edit params in config_manager.py and restart bot with args --reconfig')
        raise SystemExit
    else:
        print('loaded config successfully. run again with --reconfig if needed')

    config.set_offline_mode(offline)
    config.setup()
    minimum_loop_time = config.timesteps * 1.0 * (not offline) #This is equal to timesteps times alpha or zero if offline
    tradingbot = TradingBot(config)
    simulation = Simulation(500, False, offline, 'Normal')
    simulation2 = Simulation(500, False, offline, 'Random')
    try:
        while True:
            exec_time = tradingbot.run(simulation, simulation2)
            wait_time = max(minimum_loop_time - exec_time, 0.0)
            logging.warning('Loop execution took {} seconds. Waiting {} seconds. (It\'s safe to force shutdown now)'.format(exec_time, wait_time))
            time.sleep(wait_time)
            if offline:
                API_offline.decrease_global_lag()

    except KeyboardInterrupt:
        print('Keyboard interrupt, shutting down')
        tradingbot.perform_shutdown()

if __name__ == '__main__':
    main()