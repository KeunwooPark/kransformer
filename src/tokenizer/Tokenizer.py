# abstract class for tokenizers

from abc import ABC, abstractmethod


class Tokenizer(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def tokenize(self, text):
        pass

    @abstractmethod
    def detokenize(self, tokens):
        pass
