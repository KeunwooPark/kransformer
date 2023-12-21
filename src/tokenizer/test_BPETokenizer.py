# add project root to path
import sys
import pathlib

root = pathlib.Path(__file__).parent.parent.parent.absolute()
sys.path.append(str(root))

from src.dataloader.TextDataLoader import TextDataLoader
from src.tokenizer.BPETokenizer import BPETokenizer


def test_BPETokenizer_postfix():
    class TestDataLoader(TextDataLoader):
        def load_text_items(self, is_train=True):
            return ["fake cake", "fake fake cake"]

    data_loader = TestDataLoader()
    tokenizer = BPETokenizer(data_loader, num_merges=3)

    # the tokenizer should merge `ake`
    tokenizer.train()
    result = tokenizer.tokenize("fake cake")
    assert result == ["f", "ake</w>", "c", "ake</w>"]


def test_BPETokenizer_prefix():
    class TestDataLoader(TextDataLoader):
        def load_text_items(self, is_train=True):
            return ["nike night"]

    data_loader = TestDataLoader()
    tokenizer = BPETokenizer(data_loader, num_merges=1)

    # the tokenizer should merge `ni`

    tokenizer.train()
    print(tokenizer.token_to_id)
    result = tokenizer.tokenize("nike night")
    assert result == ["ni", "k", "e", "</w>", "ni", "g", "h", "t", "</w>"]


def test_BPETokenizer_middle():
    class TestDataLoader(TextDataLoader):
        def load_text_items(self, is_train=True):
            return ["wait and gain"]

    data_loader = TestDataLoader()
    tokenizer = BPETokenizer(data_loader, num_merges=1)

    # the tokenizer should merge `ake`
    tokenizer.train()
    result = tokenizer.tokenize("wait and gain")
    assert result == [
        "w",
        "ai",
        "t",
        "</w>",
        "a",
        "n",
        "d",
        "</w>",
        "g",
        "ai",
        "n",
        "</w>",
    ]
