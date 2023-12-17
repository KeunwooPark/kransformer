# train BPE tokenizer with Arxiv dataset

# add project root to path
import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).parent.parent.absolute()))

from src.dataloader.ArxivDataLoader import ArxivDataLoader
from src.tokenizer.BPETokenizer import BPETokenizer

data_loader = ArxivDataLoader()
tokenizer = BPETokenizer(data_loader)
tokenizer.train()
tokenizer.save("data/tokenizer/bpe_tokenizer.txt")

print(tokenizer.tokenize("hello world"))
print(tokenizer.encode("hello world"))

# test loading tokenizer
tokenizer = BPETokenizer(data_loader)
tokenizer.load("data/tokenizer/bpe_tokenizer.txt")
print(tokenizer.tokenize("tokenizer is loaded successfully"))
print(tokenizer.encode("tokenizer is loaded successfully"))
