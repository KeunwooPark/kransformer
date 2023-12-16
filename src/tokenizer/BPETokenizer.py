# implementing Sennrich et al. (2015) BPE algorithm
# https://arxiv.org/abs/1508.07909

import collections
import re
from .Tokenizer import Tokenizer


class BPETokenizer(Tokenizer):
    def __init__(self, data_loader):
        super().__init__(data_loader)
        self.num_merges = 100

    def train(self):
        text_items = self.data_loader.load_text_items()
        # text_items = ["low lower lowest newer newest", "low lower lowest"]
        word_counts = self._get_word_counts(text_items)
        for _ in range(self.num_merges):
            print(
                "BPETokenizer train iteration: " + str(_) + "/" + str(self.num_merges)
            )
            pairs = self._get_pairs(word_counts)
            if not pairs:
                break

            best = max(pairs, key=pairs.get)
            word_counts = self._merge_word_counts(best, word_counts)

        print(word_counts)
        return word_counts

    def _get_word_counts(self, text_items):
        word_counts = collections.defaultdict(int)
        words = self._split_text_items_to_words(text_items)

        for word in words:
            word_counts[word] += 1

        return word_counts

    def _split_text_items_to_words(self, text_items):
        words = []
        for text_item in text_items:
            _words = text_item.split()
            for _word in _words:
                # add spaces between each character
                # this is because to distinguish combined tokens.
                _word = " ".join(list(_word))
                # because we use space to distinguish tokens, we need to add </w> to the end of each word to distinguish words.
                _word += " </w>"
                words.append(_word)

        return words

    def _get_pairs(self, word_counts):
        pairs = collections.defaultdict(int)
        for word, count in word_counts.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[symbols[i], symbols[i + 1]] += count

        return pairs

    def _merge_word_counts(self, pair, word_counts):
        new_word_counts = collections.defaultdict(int)

        # since we put spaces between each character, the bigram should have spaces between each character, too
        bigram = re.escape(" ".join(pair))
        # this regex matches the bigram that is not surrounded by non-whitespace characters
        # which means, it matches only non-combined tokens.
        # e.g. if the bigram is "a b", it matches "a b" in "a b c", but not in "a bc"
        p = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")
        for word, count in word_counts.items():
            # if the bigram exists, merge the bigram into one token by removing the space between the characters
            w_out = p.sub("".join(pair), word)
            new_word_counts[w_out] = count

        return new_word_counts

    def tokenize(self, text):
        raise NotImplementedError

    def detokenize(self, tokens):
        raise NotImplementedError
