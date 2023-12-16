# the abastract class for loading text data

from abc import ABC, abstractmethod


class TextDataLoader(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_vocab_freq(self):
        pass
