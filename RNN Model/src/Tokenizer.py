"""
Tokenization class for neural-complexity meant to copy (very loosely) certain
functions of HuggingFace's tokenizers library. This is gross and should be
ultimately fixed up at some point...
"""
import torch
import re

from typing import List, Union, Optional

from src.data import SentenceCorpus
from src.data import sent_tokenize

class Tokenizer(SentenceCorpus):

    def __init__(self, vocab_file):
        super().__init__('.', vocab_file, interact_flag=True)
        self.model_max_length = int(1e30)
        self._unk_token = "<unk>"
        self._bos_token = "<s>"
        self._pad_token = "<pad>"

    @property
    def vocab_size(self) -> int:
        return len(self.dictionary)

    @property
    def unk_token(self) -> str:
        return str(self._unk_token)

    @property 
    def bos_token(self) -> str:
        return str(self._bos_token)

    @property
    def pad_token(self) -> str:
        return str(self._pad_token)

    @property
    def pad_token_id(self) -> Optional[int]:
        """
        :obj:`Optional[int]`: Id of the padding token in the vocabulary. Returns :obj:`None` if the token has not been
        set.
        """
        if self._pad_token is None:
            return None
        return self.dictionary.word2idx[self.pad_token]

    @property
    def unk_token_id(self):
        if self.unk_token in self.dictionary.word2idx:
            return self.dictionary.word2idx[self.unk_token]
        else:
            return None

    @property
    def bos_token_id(self):
        if self.bos_token in self.dictionary.word2idx:
            return self.dictionary.word2idx[self.bos_token]
        else:
            return None

    def __len__(self):
        return self.vocab_size

    def encode(self, line, add_space_before_punct_symbol=True, lower=True,
            remove_trailing_spaces=True):

        if lower:
            line = line.lower()

        if remove_trailing_spaces:
            line = line.strip()

        if add_space_before_punct_symbol:
            punct = "!\"#$%&'()*+,./:;-=?@[\]^_`{|}~"
            #add space before punct
            line = line.translate(str.maketrans({key: " {0}".format(key) for key in punct}))

            #break things like "farm-house" into "farm - house" and "and/or" into "and / or" careful here
            punct = "/-"
            #add space before punct
            line = line.translate(str.maketrans({key: "{0} ".format(key) for key in punct}))

            #remove double spaces
            line = re.sub('\s{2,}', ' ', line)

        sentences = sent_tokenize(line)
        output = []
        for x, sent in enumerate(sentences):
            sent = sent.split(' ')
            if x == 0:
                sent = [self.bos_token] + sent 
            test_sent = ' '.join(sent + ['the'])
            if len(sent_tokenize(test_sent)) != 1:
                sent += [self.bos_token]

            output += list(self.convert_tokens_to_ids(sent))
        return output

    def convert_ids_to_tokens(self, ids):
        if type(ids) != list:
            ids = [ids]
        return self.decode(ids)

    def decode(self, ids):
        words = list(map(lambda x: self.dictionary.idx2word[x], ids))
        return words

    def convert_tokens_to_ids(self, 
            tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        if tokens is None:
            return None
        if isinstance(tokens, str):
            return self._convert_token_to_id(tokens)
        ids = []
        for token in tokens:
            ids.append(self._convert_token_to_id(token))
        return ids

    def _convert_token_to_id(self, token):
        if token in self.dictionary.word2idx:
            return self.dictionary.word2idx[token]
        return self.unk_token_id
