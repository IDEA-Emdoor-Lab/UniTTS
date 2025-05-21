# Copyright (c) 2021, EleutherAI
# This file is based on code by the authors denoted below and has been modified from its original version.
#
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Megatron tokenizers."""

from abc import ABC
from abc import abstractmethod
import os

from tokenizers import Tokenizer
from transformers import AutoTokenizer
import numpy as np
import sentencepiece as spm
from typing import List, Union


def build_tokenizer(args):
    """Initialize tokenizer."""
    if args.rank == 0:
        print("> building {} tokenizer ...".format(args.tokenizer_type), flush=True)

    # Select and instantiate the tokenizer.
    if args.tokenizer_type.lower() == "LlamaTokenizer".lower():
        assert args.vocab_file is not None
        tokenizer = LlamaTokenizer(args.vocab_file)
    else:
        raise NotImplementedError(
            "{} tokenizer is not " "implemented.".format(args.tokenizer_type)
        )

    # Add vocab size.
    args.padded_vocab_size = _vocab_size_with_padding(tokenizer.vocab_size, args)

    return tokenizer


def build_qwen_tokenizer(args):
    """Initialize tokenizer."""
    if args.rank == 0:
        print("> building {} tokenizer ...".format(args.tokenizer_type), flush=True)

    tokenizer = QWenTokenizer(args.load)

    return tokenizer


def _vocab_size_with_padding(orig_vocab_size, args):
    """Pad vocab size so it is divisible by model parallel size and
    still having GPU friendly size."""

    after = orig_vocab_size
    multiple = args.make_vocab_size_divisible_by * args.model_parallel_size
    while (after % multiple) != 0:
        after += 1
    if args.rank == 0:
        print(
            " > padded vocab (size: {}) with {} dummy tokens "
            "(new size: {})".format(orig_vocab_size, after - orig_vocab_size, after),
            flush=True,
        )
    return after


class AbstractTokenizer(ABC):
    """Abstract class for tokenizer."""

    def __init__(self, name):
        self.name = name
        super().__init__()

    @property
    @abstractmethod
    def vocab_size(self):
        pass

    @property
    @abstractmethod
    def vocab(self):
        """Dictionary from vocab text token to id token."""
        pass

    @property
    @abstractmethod
    def inv_vocab(self):
        """Dictionary from vocab id token to text token."""
        pass

    @abstractmethod
    def tokenize(self, text):
        pass

    def detokenize(self, token_ids):
        raise NotImplementedError(
            "detokenizer is not implemented for {} " "tokenizer".format(self.name)
        )

    @property
    def cls(self):
        raise NotImplementedError(
            "CLS is not provided for {} " "tokenizer".format(self.name)
        )

    @property
    def sep(self):
        raise NotImplementedError(
            "SEP is not provided for {} " "tokenizer".format(self.name)
        )

    @property
    def pad(self):
        raise NotImplementedError(
            "PAD is not provided for {} " "tokenizer".format(self.name)
        )

    @property
    def eod(self):
        raise NotImplementedError(
            "EOD is not provided for {} " "tokenizer".format(self.name)
        )

    @property
    def mask(self):
        raise NotImplementedError(
            "MASK is not provided for {} " "tokenizer".format(self.name)
        )


class LlamaTokenizer(AbstractTokenizer):
    """Designed to LlamaTokenizer."""
    def __init__(self, vocab_file, fast=True):
        name = "LlamaTokenizer"
        super().__init__(name)
        if os.path.isfile(vocab_file):
            vocab_file = os.path.dirname(vocab_file)
        print('===================', flush=True)
        print(vocab_file, flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(vocab_file, trust_remote_code=True)
        sp_tokens = {'additional_special_tokens': ['<human>', '<bot>', '<ur_eod>']}
        self.tokenizer.add_special_tokens(sp_tokens)
        self.pad_id = self.tokenizer.pad_token_id
        self.bos_id = self.tokenizer.bos_token_id
        self.eod_id = self.tokenizer.eos_token_id
        self.sep_id = self.tokenizer.sep_token_id
        self.human_id = self.tokenizer.convert_tokens_to_ids('<human>')
        # assert self.human_id
        self.bot_id = self.tokenizer.convert_tokens_to_ids('<bot>')
        # assert self.bot_id
        self.ur_id = self.tokenizer.convert_tokens_to_ids('<ur_eod>')

    @property
    def vocab_size(self):
        return len(self.tokenizer)

    @property
    def inv_vocab(self):
        return self.tokenizer.decoder

    @property
    def vocab(self):
        return self.tokenizer.get_vocab()

    @property
    def eod(self):
        return self.eod_id

    @property
    def bos(self):
        return self.bos_id

    def tokenize(self, text: str):
        return self.tokenizer.encode(text)

    def detokenize(self, token_ids):
        return self.tokenizer.decode(token_ids)


class QWenTokenizer(AbstractTokenizer):
    """Designed to LlamaTokenizer."""
    def __init__(self, tokenizer_config, fast=True):
        name = "QWenTokenizer"
        super().__init__(name)

        print(f'Loading {name} configs...')
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_config)
        self.pad_id = self.tokenizer.pad_token_id
        self.eos_id = self.tokenizer.eos_token_id
        self.bos_id = self.eos_id
        print(f'Pad-ID: {self.pad_id}\nBOS-ID: {self.bos_id}\nEOS-ID: {self.eos_id}')

        self.begin_of_message = self.tokenizer.convert_tokens_to_ids('<|im_start|>')
        self.end_of_message = self.tokenizer.convert_tokens_to_ids('<|im_end|>')
        print(f'Mesage begin: {self.begin_of_message}\nMessage end: {self.end_of_message}')

        self.begin_of_audio = self.tokenizer.convert_tokens_to_ids('<|beginofaudio|>')
        self.end_of_audio = self.tokenizer.convert_tokens_to_ids('<|endofaudio|>')
        print(f'Begin Audio: {self.begin_of_audio}\nEnd Audio: {self.end_of_audio}')

        self.inter_audio_begin = self.tokenizer.convert_tokens_to_ids('<|inter_audio_begin|>')
        self.inter_audio_end = self.tokenizer.convert_tokens_to_ids('<|inter_audio_end|>')
        print(f'Inter Begin Audio: {self.inter_audio_begin}\nInter End Audio: {self.inter_audio_end}')

        self.cot_begin = self.tokenizer.convert_tokens_to_ids('<|cot_begin|>')
        self.cot_end = self.tokenizer.convert_tokens_to_ids('<|cot_end|>')
        print(f'COT Begin: {self.cot_begin}\nCOT End: {self.cot_end}')

        self.tool_begin = self.tokenizer.convert_tokens_to_ids('<tool_call>')
        self.tool_end = self.tokenizer.convert_tokens_to_ids('</tool_call>')
        print(f'Tool Begin: {self.tool_begin}\nTool End: {self.tool_end}')

    @property
    def vocab_size(self):
        return len(self.tokenizer)

    @property
    def inv_vocab(self):
        return self.tokenizer.decoder

    @property
    def vocab(self):
        return self.tokenizer.get_vocab()

    @property
    def eod(self):
        return self.eod_id

    @property
    def bos(self):
        return self.bos_id

    def tokenize(self, text: str):
        return self.tokenizer.encode(text)

    def detokenize(self, token_ids):
        return self.tokenizer.decode(token_ids)
    

if __name__ == '__main__':
    qwen = QWenTokenizer(tokenizer_config='/cognitive_comp/ccnl_common_data/wangrui/audio-text-models/qwen_models/Qwen2.5-7B-Codec0927-S204000-AEdivided100')
