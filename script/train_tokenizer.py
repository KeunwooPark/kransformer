# train BPE tokenizer with Arxiv dataset

# add project root to path
import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).parent.parent.absolute()))

from src.dataloader.ArxivDataLoader import ArxivDataLoader
from src.tokenizer.BPETokenizer import BPETokenizer

data_loader = ArxivDataLoader()
tokenizer = BPETokenizer(data_loader, num_merges=1000)
tokenizer.train()
tokenizer.save("data/tokenizer/bpe_tokenizer.txt")

print("tokenize", tokenizer.tokenize("hello world"))
print("encode", tokenizer.encode("hello world"))
print("decode", tokenizer.decode(tokenizer.encode("hello world")))

# test loading tokenizer
tokenizer = BPETokenizer(data_loader)
tokenizer.load("data/tokenizer/bpe_tokenizer.txt")
print("tokenize", tokenizer.tokenize("tokenizer is loaded successfully."))
print("encode", tokenizer.encode("tokenizer is loaded successfully."))
print("decode", tokenizer.decode(tokenizer.encode("tokenizer is loaded successfully.")))
