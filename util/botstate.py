import abc
class State(abc.ABCMeta):

    @abc.abstractmethod
    def message(self):
        pass

class RUNNING(State):
    def message(self):
        print('Bot is running')

class PAUSED(State):
    def message(self):
        print('Bot is paused')

class STOPPED(State):
    def message(self):
        print('Bot has stopped')