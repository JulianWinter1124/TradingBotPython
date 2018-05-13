import bot.parser
from util.printer import eprint

if __name__ == '__main__':
    eprint('start')
    parser = bot.parser.Parser()
    parser.start()
