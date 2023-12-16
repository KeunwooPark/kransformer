# testing tokenizer

from src.dataloader.ArxivDataLoader import ArxivDataLoader
from src.tokenizer.BPETokenizer import BPETokenizer

data_loader = ArxivDataLoader()
tokenizer = BPETokenizer(data_loader)
tokenizer.train()
