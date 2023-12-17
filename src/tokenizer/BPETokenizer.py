# implementing Sennrich et al. (2015) BPE algorithm
# https://arxiv.org/abs/1508.07909

import collections
import re
from .Tokenizer import Tokenizer
import pathlib


class BPETokenizer(Tokenizer):
    def __init__(self, data_loader, num_merges=1000):
        super().__init__(data_loader)
        self.num_merges = num_merges

    def train(self):
        text_items = self.data_loader.load_text_items()

        word_counts = self._get_word_counts(text_items)
        for _ in range(self.num_merges):
            print(
                "BPETokenizer train iteration: " + str(_) + "/" + str(self.num_merges)
            )
            pair_counts = self._get_token_pair_counts(word_counts)
            if not pair_counts:
                break

            best = max(pair_counts, key=pair_counts.get)
            word_counts = self._merge_word_counts(best, word_counts)

        self.token_to_id = self._create_token_to_id(word_counts)
        self.id_to_token = self._create_id_to_token(self.token_to_id)

    def save(self, path):
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for token, id_and_count in self.token_to_id.items():
                id = id_and_count["id"]
                count = id_and_count["count"]
                f.write(token + "\t" + str(id) + "\t" + str(count) + "\n")

    def load(self, path):
        self.token_to_id = {}
        with open(path, "r") as f:
            for line in f:
                token, id, count = line.strip().split("\t")
                self.token_to_id[token] = {"id": int(id), "count": int(count)}
        self.id_to_token = self._create_id_to_token(self.token_to_id)

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

    def _get_token_pair_counts(self, word_counts):
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

    def _create_token_to_id(self, word_counts):
        token_to_id = {}

        for word, count in word_counts.items():
            tokens = word.split()
            for token in tokens:
                if token not in token_to_id:
                    # set the id to 0. we will change the id later.
                    token_to_id[token] = {"id": 0, "count": count}

                else:
                    token_to_id[token]["count"] += count

        # sort the tokens and assign the id
        # we want to have the similar tokens to have similar ids
        # so we sort the tokens by the characters in the token
        # i think it will help the model to learn the embeddings better

        sorted_tokens = sorted(token_to_id.keys())
        cur_id = 0
        for token in sorted_tokens:
            token_to_id[token]["id"] = cur_id
            cur_id += 1

        return token_to_id

    def _create_id_to_token(self, token_to_id):
        id_to_token = {}
        for token, id_and_count in token_to_id.items():
            id = id_and_count["id"]
            id_to_token[id] = token

        return id_to_token

    def tokenize(self, text):
        words = self._split_text_items_to_words([text])
        tokens = []
        for word in words:
            tokens += self._tokenize_word(word)

        return tokens

    def _tokenize_word(self, word):
        # merge tokens until the tokens cannot be merged anymore
        # the merge process is the same as the training process
        # it creates all the possible bigrams and merge the bigram that has the highest count in the word_counts

        word_count = {word: 1}
        while True:
            pair_counts = self._get_token_pair_counts(word_count)

            # for all the pairs, find the pair with the highest count in the token_to_id
            best_pair = None
            for pair in pair_counts:
                bigram = "".join(pair)
                if bigram in self.token_to_id:
                    if best_pair is None:
                        best_pair = pair
                        continue
                    else:
                        best_bigram = "".join(best_pair)
                        if (
                            self.token_to_id[bigram]["count"]
                            > self.token_to_id[best_bigram]["count"]
                        ):
                            best_pair = pair

            if best_pair is None:
                break

            word_count = self._merge_word_counts(best_pair, word_count)

        # after the merge process, the word_count should have only one key
        # which is the merged token
        # return the merged token
        tokenized_word = list(word_count.keys())[0]
        return tokenized_word.split()

    def encode(self, text):
        tokens = self.tokenize(text)
        ids = []
        for token in tokens:
            ids.append(self.token_to_id[token]["id"])
        return ids

    def decode(self, ids):
        tokens = []
        for id in ids:
            tokens.append(self.id_to_token[id])
        return self.detokenize(tokens)

    def detokenize(self, tokens):
        text = "".join(tokens)
        text = text.replace("</w>", " ")
        return text
