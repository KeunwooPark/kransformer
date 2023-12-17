# abstract class for tokenizers

from abc import ABC, abstractmethod
from src.dataloader.TextDataLoader import TextDataLoader


class Tokenizer(ABC):
    def __init__(self, data_loader: TextDataLoader):
        self.data_loader = data_loader
        pass

    @abstractmethod
    def train(self, corpus):
        pass

    @abstractmethod
    def save(self, path):
        pass

    @abstractmethod
    def load(self, path):
        pass

    @abstractmethod
    def tokenize(self, text):
        pass

    @abstractmethod
    def detokenize(self, tokens):
        pass

    @abstractmethod
    def encode(self, text):
        pass

    @abstractmethod
    def decode(self, tokens):
        pass
