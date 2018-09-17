# Implementing a Cryptocurrency Trading Bot

This is the Code for the bachelor thesis of Julian Winter

## Installation

Go to https://www.ta-lib.org/hdr_dw.html and download the necessary files for Linux.

Follow the install instructions (mostly just copying files).

Change to the bot directory with:

>cd [your path]/TradingBotPython

Make a new Python environment and run the following command to install all libraries:

> pip install -r requirements.txt

You can also install all libraries manually by reading the section **3.2 Programmiersprache und Bibliotheken** in the bachelor thesis and installing them one by one.

For example:

>pip install numpy

## Configurating

If desired, change the settings in /bot/config_manager.py in the *init_vars* method .

To apply the changes run the bot with the --reconfig argument described in [Running the bot](#running-the-bot).


## Running the bot

Change to the bot directory with:

>cd [your path]/TradingBotPython

Assuming you already configured the bot, you can run it with the optional arguments:

>python __main__.py [--offline, --reconfig, --log] [--help]

# Examples\

>python __main__.py --help

>python __main__.py --offline --reconfig

>python __main__.py --offline

>python __main__.py --log 30





