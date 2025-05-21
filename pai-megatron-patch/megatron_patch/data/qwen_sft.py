# Copyright (c) 2023 Alibaba PAI Team.
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

import numpy as np
import io
import copy
import json
import torch
try:
    from megatron import get_args
except:
    from megatron.training import get_args
from datasets import load_dataset
from tqdm import tqdm
import os

from megatron_patch.tokenizer import get_tokenizer

import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

class QwenSFTDataset(torch.utils.data.Dataset):
    """A class for processing a LLama text dataset"""
    def process_line(self, line, max_padding_length, tokenizer, IGNORE_INDEX):
        data_dict = json.loads(line.strip())
        position_ids = np.arange(max_padding_length)
        attention_mask = np.zeros((max_padding_length, max_padding_length))

        for sidx in data_dict["sample_idx"]:
            attention_mask[sidx[0] : sidx[1], sidx[0] : sidx[1]] = np.tril(
                np.ones((sidx[1] - sidx[0], sidx[1] - sidx[0]))
            )[: max_padding_length, : max_padding_length]

            position_ids[sidx[0] : sidx[1]] = np.arange(sidx[1] - sidx[0])[
                : max_padding_length
            ]

        attention_mask.dtype = "int64"
        
        data_dict["text"] = data_dict["text"] + [tokenizer.pad_token_id] * max_padding_length
        data_dict["loss_mask"] = data_dict["loss_mask"] + [IGNORE_INDEX] * max_padding_length
        if sum(data_dict["loss_mask"][: max_padding_length - 1]) == 0: # 防止截断后mask全为0，导致loss=nan
            data_dict["loss_mask"][max_padding_length - 2] = 1
        
        sample = {
            "input_ids": np.array(
                data_dict["text"][: max_padding_length] + [tokenizer.pad_token_id], dtype=np.int64
            ),
            "loss_mask": np.array(
                data_dict["loss_mask"][: max_padding_length], dtype=np.int64
            ),
            "attention_mask": attention_mask[
                : max_padding_length, : max_padding_length
            ],
            "position_ids": position_ids[: max_padding_length],
        }
        return sample


    def __init__(self, path, max_padding_length, split='train'):
        self.tokenizer = get_tokenizer()
        assert hasattr(self.tokenizer, 'apply_chat_template'), \
            "The LLama-SFT-Raw Dataset is valid for tokenizers with chat template, please provide a template."
        self.IGNORE_INDEX = self.tokenizer.pad_token_id
        self.max_padding_length = max_padding_length
        self.samples = []
        print(f'train dataset path:{path}', flush=True)
        n_samples = 0
        for filename in path:
            if os.path.exists(filename):
                with open(filename) as f:
                    for line in tqdm(f, desc=filename):
                        data_dict = json.loads(line.strip())
                        
                        sample = {
                            "input_ids": 
                                data_dict["text"]
                            ,
                            "loss_mask": 
                                data_dict["loss_mask"]
                            ,
                            "sample_idx": 
                                data_dict["sample_idx"]
                        }

                        self.samples.append(sample)
                        n_samples += 1
                        if n_samples % 10000 == 0:
                            torch.distributed.barrier()
                        if n_samples == 160000:
                            print(n_samples)
                            break
            n_samples = 0

        print('  >> total number of samples: {}'.format(len(self.samples)))

    def _make_r_io_base(self, f, mode: str):
        if not isinstance(f, io.IOBase):
            f = open(f, mode=mode, encoding='utf-8')
        return f

    def jload(self, f, mode='r'):
        """
        Load a .json file into a dictionary.
        Args:
            f: The file object or string representing the file path.
            mode: The mode in which to open the file (e.g., 'r', 'w', 'a').
        Returns:
            A dictionary containing the contents of the JSON file.
        """
        f = self._make_r_io_base(f, mode)
        jdict = json.load(f)
        f.close()
        return jdict

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        raw_sample = self.samples[idx]
        # print(f"raw_sample[input_ids] shape is: {raw_sample['loss_mask']}", flush=True)
        
        return raw_sample

    def preprocess(self, examples):
        """
        Preprocess the data by tokenizing.
        Args:
            sources (List[str]): a list of source strings
            targets (List[str]): a list of target strings
            tokenizer (Tokenizer): a tokenizer object used for tokenization
        Returns:
            dict: a dictionary containing the input_ids and labels for the examples
        """


        datas = []
        if 'instruction' in examples:
            datas = [ examples['instruction']]
        elif 'query' in examples:
            datas = [ examples['query']]
        else:
            raise ValueError('Cannot find key instruction or query!')

        if 'input' in examples:
            datas.append(examples['input'])

        if 'output' in examples:
            datas.append(examples['output'])
        elif 'content' in examples:
            datas.append(examples['content'])
        elif 'response' in examples:
            datas.append(examples['response'])
        else:
            raise ValueError('Cannot find output key `output`, `content` or `response`!')
        
        input_ids = []
        labels = []
        for data in zip(*datas):
            text = [
                {'role': 'user', 'content': ''.join(data[:-1])},
                {'role': 'assistant', 'content': data[-1]}
            ]
            source = self.tokenizer.apply_chat_template(text[:-1])
            full = self.tokenizer.apply_chat_template(text)

            for t1, t2 in zip(source, full):
                assert t1 == t2, "The user input_ids are not a prefix of the full input_ids! Please check the template."
            
            if len(source) >= self.max_padding_length:
                continue

            if len(full) >= self.max_padding_length:
                full = full[:self.max_padding_length]
            
            if self.max_padding_length > len(full):
                full = full + [self.IGNORE_INDEX] * (self.max_padding_length - len(full))
            
            # NOTE: in get_batch_on_this_tp_rank_original, tokens use [:-1] and labels use [1:]
            # we add an extra token to use the old api
            # TODO: update get_batch_on_this_tp_rank_original and replace the following line with
            # label = [self.IGNORE_INDEX] * (len(source) - 1) + full[len(input_ids):] + [self.IGNORE_INDEX]
            full = full + [self.IGNORE_INDEX]
            label = [self.IGNORE_INDEX] * len(source) + full[len(source):]

            input_ids.append(full)
            labels.append(label)

        return dict(input_ids=input_ids, labels=labels)

    def tokenize(self, strings, tokenizer):
        """
        This API is only for consistency and not used in the SFT dataset.
        """

        tokenized_list = [
            tokenizer(
                text,
                return_tensors='pt',
                padding='max_length',
                max_length=self.max_padding_length,
                truncation=True,
                add_special_tokens=False
            ) for text in strings
        ]
        input_ids = labels = [
            tokenized.input_ids[0] for tokenized in tokenized_list
        ]
        input_ids_lens = labels_lens = [
            (tokenized.input_ids != tokenizer.pad_token_id).sum().item()
            for tokenized in tokenized_list
        ]
        return dict(
            input_ids=input_ids,
            labels=labels,
            input_ids_lens=input_ids_lens,
            labels_lens=labels_lens,
        )

    def gpt_convert_example_to_feature(self, sample):
        """
        Convert a single sample containing input_id, label and loss_mask into a format suitable for GPT training.
        """
        input_ids, labels = sample
        train_sample = {
            'input_ids': input_ids,
            'labels': labels
        }

        return train_sample
