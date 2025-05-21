# Copyright (c) 2023 Alibaba PAI and Nvidia Megatron-LM Team.
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

import argparse
import math
import json
import os
import sys
import re
from tqdm import tqdm
import random
import numpy as np
from multiprocessing import Pool
import multiprocessing
from functools import partial

from utils.tokenizer import build_qwen_tokenizer, QWenTokenizer # type: ignore



sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
import time
import gzip
import multiprocessing

from megatron.core.datasets import indexed_dataset

from megatron_patch.tokenizer import build_tokenizer

scaling_factor=1000000

class Encoder(object):
    def __init__(self, args):
        self.args = args

        self.conv_template = "<|im_start|>{role_name}\n{message}<|im_end|><|endoftext|>\n"

        self.n_single_print = 0

        self.is_concat_print = True
        
    def initializer(self):
        # Use Encoder class as a container for global data
        Encoder.tokenizer: QWenTokenizer = build_qwen_tokenizer(self.args)
        Encoder.seq_length = self.args.seq_length
    
    def is_complete_sentence(self, item):
        """判断句子是否为一个完整的句子 

        Args:
            item (_type_): _description_
        """
        pattern = r"[。！？；.!?;]$"
        is_complete_sen = True
        for text in item['prompt']:
            match = re.search(pattern, text)
            if not match:
                is_complete_sen = False

        return is_complete_sen
    
    def convert_audioID2str(self, audio_tokens) -> str:
        audio_token_str = ''
        for audio_token in audio_tokens:
            audio_token_str += f'<|g0r0_{audio_token}|>'

        return audio_token_str

    def encode(self, item):
        item = json.loads(item)
        if 'dialog' in item:
            dialoge = item['dialog']
        else:
            dialoge = item['dialoge']
        tokens = []
        loss_mask = []
        sample_in_str = ''

        # 增加role dialog膨胀的对话数据集
        role_tokens = []
        role_loss_mask = []
        role_sample_in_str = ''
        dialoge = json.loads(dialoge) if isinstance(dialoge, str) else dialoge
        # print(f'dialogeis:{dialoge}', flush=True)
        for index, conv_turn in enumerate(dialoge):
            role_name = None
            role_type = None
            if 'type' in conv_turn:
                role_type = conv_turn['type']
            elif 'role' in conv_turn: 
                role_type = conv_turn['role']
    
            if 'name' in conv_turn:
                role_name = conv_turn['name']
            if role_name is None:
                role_name = role_type
            if role_type is None:
                raise ValueError(f'can not find role type!')

            if 'system' in role_type:
                role_name = 'system'
                cal_loss = False
            elif role_type == 'assistant':
                cal_loss = True
            elif role_type == 'user':
                cal_loss = False
            else:
                raise ValueError(f'{role_type} not supported yet!')
            if 'cal_loss' in conv_turn:
                cal_loss = conv_turn['cal_loss']

            message = ''
            message_role = ''
            n_segs = len(conv_turn['content_list'])
            for i, conv_seg in enumerate(conv_turn['content_list']):
                content: str = conv_seg['content']
                if i == 0:
                    content = content.lstrip()
                if i == n_segs - 1:
                    content = content.rstrip()

                content_type = conv_seg['type']
                if content_type == 'text':
                    message += content
        
                    message_role += content
                elif content_type == 'audio':
                    audio_token_str = self.convert_audioID2str(conv_seg['audio_infos']['audio_tokens'])
                    conv_seg['audio_infos']['audio_tokens'] = []
                    audio_str = f'<|beginofaudio|>{audio_token_str}<|endofaudio|>'

                    if role_type != 'assistant':
                        message_role += audio_str

                    if '<|audio|>' in content:
                        content = content.replace('<|audio|>', audio_str)
                        message += content
                    else:
                        print(content)
                        message += audio_str
                elif content_type == 'audio_text':
                    audio_token_str = self.convert_audioID2str(conv_seg['audio_infos']['audio_tokens'])
                    conv_seg['audio_infos']['audio_tokens'] = []
                    audio_str = f'<|inter_audio_begin|>{audio_token_str}<|inter_audio_end|>'
                    
                    if '<|audio|>' in content:
                        content = content.replace('<|audio|>', audio_str)
                    else:
                        raise ValueError()
                    message += content

                    if role_type == 'assistant':
                        message_role += conv_seg['audio_infos']['text']
                    else:
                        message_role += content
                elif content_type == 'cot':
                    cot_str = f'<|cot_begin|>{content}<|cot_end|>\n'
                    message += cot_str
                    message_role += cot_str
                elif 'tool' in content_type:
                    tool_str = f'<tool_call>{content}</tool_call>\n'
                    message += tool_str
                    message_role += tool_str
                else:
                    raise ValueError(f'{content_type} not supported yet~')
            # 完整的信息 
            turn_in_str = self.conv_template.format(
                role_name=role_name, 
                message=message)
            tokens_t = self.tokenizer.tokenize(turn_in_str.strip())
            tokens.extend(tokens_t)
            # 为了归一化sft的answer长度，因此loss mask要除以answer的长度
            tokens_len = len(tokens_t[:-1])
            loss_mask_t = [1 / tokens_len if cal_loss else 0 for _ in tokens_t[:-1]] + [0]
            loss_mask_t = (np.array(loss_mask_t) * scaling_factor).astype(int).tolist()
            # print(loss_mask_t)

            # loss_mask_t = [1 if cal_loss else 0 for _ in tokens_t[:-1]] + [0]
            loss_mask.extend(loss_mask_t)
            sample_in_str += turn_in_str

            # 如果是role dialog类型，则进行膨胀对话
            if item['task_category'] == 'role_dialog':
                role_turn_in_str = self.conv_template.format(
                    role_name=role_name, 
                    message=message_role)
                role_tokens_t = self.tokenizer.tokenize(role_turn_in_str.strip())
                # role_loss_mask_t = [1 if cal_loss and role_type == 'assistant' else 0 for _ in role_tokens_t[:-1]] + [0]
                
                if role_type == 'assistant': 
                    dialog_role_tokens = []
                    dialog_role_tokens.extend(role_tokens)
                    dialog_role_tokens.extend(tokens_t)

                    dialog_role_loss_mask = []
                    dialog_role_loss_mask.extend(role_loss_mask)
                    dialog_role_loss_mask.extend(loss_mask_t)

                    dialog_role_sample_in_str = role_sample_in_str
                    dialog_role_sample_in_str += turn_in_str

                    # 构造出来一轮对话
                    role_tokens_list = [dialog_role_tokens]
                    role_loss_mask_list = [dialog_role_loss_mask]
                    num_of_tokens = len(role_tokens_list[0])
                    not_concat_flag = True
                    is_complete_sentence = True
                    ret = {
                        "tokens": role_tokens_list, 
                        "loss_mask": role_loss_mask_list, 
                        "task_category": item['task_category'],
                        'not_concat_flag': not_concat_flag, 
                        "num_of_tokens": num_of_tokens, 
                        'sample_info': None,
                        'is_complete_sentence': is_complete_sentence
                    }
                    if self.n_single_print < 4:
                        print(f'第{self.n_single_print}条：\n{dialog_role_sample_in_str}\n\n')
                        self.n_single_print += 1
                    yield ret

                    #import sys
                    #sys.exit(0)
                if self.args.role_dialog_context_type_text:
                    role_tokens.extend(role_tokens_t)   
                    role_loss_mask_t_prefix = [ 0 for _ in role_tokens_t]
                    role_loss_mask.extend(role_loss_mask_t_prefix)
                    role_sample_in_str += role_turn_in_str
                else:
                    role_tokens.extend(tokens_t)   
                    role_loss_mask_t_prefix = [ 0 for _ in tokens_t]
                    role_loss_mask.extend(role_loss_mask_t_prefix)
                    role_sample_in_str += turn_in_str

        if item['task_category'] != 'role_dialog':
            assert len(tokens) == len(loss_mask)
            if self.n_single_print < 10:
                print(f'第{self.n_single_print}条：\n{sample_in_str}\n\n')
                if self.n_single_print == 0:
                    
                    for tid, lmask in zip(tokens, loss_mask):
                        print(f'Token id: {tid}  Loss Mask: {lmask}')
                self.n_single_print += 1

            tokens = [tokens]
            loss_mask = [loss_mask]
            num_of_tokens = len(tokens[0])
            not_concat_flag = True
            is_complete_sentence = True
            ret = {
                "tokens": tokens, 
                "loss_mask": loss_mask, 
                "task_category": item['task_category'],
                'not_concat_flag': not_concat_flag, 
                "num_of_tokens": num_of_tokens, 
                'sample_info': None,
                'is_complete_sentence': is_complete_sentence
            }
            
            yield ret
    
    def encode_blocked(self, datas):
        return list(self.encode(datas))

    def concat(self, encode_rs):
        """针对单轮数据做样本拼接

        Args:
            encode_rs (_type_): _description_

        Returns:
            _type_: _description_
        """
        

        idx = 0              #  id index
        sample_list = []     #  返回的结果
        
        while idx < len(encode_rs):
            ids = {} 
            lens = {}

            tokens = []
            loss_mask = []
            sample_idx = []
            cur = 0 # 截止到目前token的数量

            while len(tokens) < Encoder.seq_length and idx < len(encode_rs):
                
                encode_result = encode_rs[idx]

                if (len(tokens) + encode_result['num_of_tokens']) > Encoder.seq_length and len(tokens) != 0:
                    break

                # 异常判断，防止出错
                if len(encode_result['tokens']) < 1 or len(encode_result['loss_mask']) < 1 or len(encode_result['tokens'][0]) != len(encode_result['loss_mask'][0]):
                    idx += 1
                    continue

                # 处理lm数据
                sample_tokens = encode_result['tokens'][0]
                sample_loss_mask = encode_result['loss_mask'][0]
                
                if len(tokens) > 0:
                    sample_idx.append([cur, min(len(tokens), Encoder.seq_length)])
                    cur = len(tokens)

                assert len(sample_tokens) == len(sample_loss_mask)
                #print(sample_tokens)
                tokens.extend(sample_tokens)
                loss_mask.extend(sample_loss_mask)
                
                idx += 1
                #cur += len(sample_tokens)

            # 添加上最后一步的sample tokens
            if cur < len(tokens):
                sample_idx.append([cur, min(len(tokens), Encoder.seq_length)])

            # 最后一步是添加上bos token
            tokens = tokens + [Encoder.tokenizer.eos_id] * Encoder.seq_length
            loss_mask = loss_mask + [0] * Encoder.seq_length
            #sample = {
            #        "text": tokens[:Encoder.seq_length],
            #        "loss_mask": loss_mask[:Encoder.seq_length],
            #        "sample_idx": sample_idx,
            #    }
            ids['text'] = tokens[:Encoder.seq_length] + loss_mask[:Encoder.seq_length]
            
            lens['text'] = [len(tokens[:Encoder.seq_length]) * 2]
            sample_list.append([ids, lens, sample_idx, len(json.dumps(ids))])

        if self.is_concat_print:
            print(self.is_concat_print)
            n0_sample = sample_list[0][0]
            sample_idx = sample_list[0][2]
            # print(n0_sample)
            tokens = n0_sample['text'][:len(n0_sample['text'])//2]
            loss_mask = n0_sample['text'][len(n0_sample['text'])//2:]
            sample_str = self.tokenizer.detokenize(tokens)
            print(f'***** Concated Sample in string:\n{sample_str}')
            print(f'Token {self.tokenizer.eos_id} means <endoftext>\nToken {self.tokenizer.begin_of_message} means <|im_start|>\nToken {self.tokenizer.end_of_message} means <|im_end|>')
            end_idxs = [ele[1] for ele in sample_idx]
            print_tokens = self.tokenizer.tokenizer.convert_ids_to_tokens(tokens)
            for i, (tid, lmask, print_token) in enumerate(zip(tokens, loss_mask, print_tokens)):
                if i + 1 in end_idxs:
                    print(f'Line{i} Token ID: {tid} Loss Mask: {lmask} decode token: {print_token}**Sample End**')
                elif i in end_idxs:
                    print(f'Line{i} Token ID: {tid} Loss Mask: {lmask} decode token: {print_token}**Sample Begin**')
                else:
                    print(f'Line{i} Token ID: {tid} Loss Mask: {lmask} decode token: {print_token}')
            self.is_concat_print = False
        
        return sample_list

    def concat_sample_idx(self, sample_list, save_apth, sample_ids_padding_length = 100):
        max_len = 0
        new_sample_list = []
        # 找到最大的sample数值
        for sample in sample_list:
            if len(sample[2][0]) > max_len:
                max_len = len(sample[2][0]) * 2
        
        max_len = sample_ids_padding_length

        # 拼接sample_idx
        for sample in sample_list:
        
            ids, lens, sample_idx, ids_bytes = sample
            sample_idx_one_dim = []
            for idx in sample_idx:
                sample_idx_one_dim.extend(idx)
                
            sample_idx_one_dim = sample_idx_one_dim + [-100] * (max_len - len(sample_idx_one_dim))
            ids['text'] = ids['text'] + sample_idx_one_dim
            lens['text'] = [lens['text'][0] + len(sample_idx_one_dim)]

            # print(f"text tokens is: {ids['text']}")
            # print(f"text tokens is: {len(ids['text'])}")
            # print(f"text tokens is: {lens['text']}")

            new_sample_list.append([ids, lens, ids_bytes])
        # 生成一个保存的文件，存储当前数据集，sample idx的文件
        with open(save_apth, 'w') as f:
            f.write(str(max_len))
        
        return new_sample_list

   
class Partition(object):
    def __init__(self, args, workers):
        self.args = args
        self.workers = workers
        self.args = get_args()

    def print_processing_stats(self, count, proc_start, total_bytes_processed):
        if count % self.args.log_interval == 0:
            current = time.time()
            elapsed = current - proc_start
            mbs = total_bytes_processed/elapsed/1024/1024
            print(f"Processed {count} documents",
                  f"({count/elapsed} docs/s, {mbs} MB/s).",
                  file=sys.stderr)
    def process_doc(self, doc, encoder):
        """
        处理单个文档的函数，调用 encoder.encode 并返回结果列表
        """
        print(f'doc is:{doc}', flush=True)
        return list(encoder.encode(doc))
    def process_json_file(self, file_name):
        input_file_name, output_prefix = file_name
        print("Opening", input_file_name)

        # json or jsonl
        try:
            with open(input_file_name, 'r', encoding='utf-8') as f:
                fin = json.load(f)
        except Exception:
            fin = []
            with open(input_file_name, 'r', encoding='utf-8') as f:
                fin = [json.loads(d) for d in f.readlines()]
        
        assert isinstance(fin, list)
        # NOTE: each item in fin is a group (dict / list[dict]) of samples may be packed together
    
        startup_start = time.time()
        encoder = Encoder(self.args)
        if self.args.sequence_packing:
            # collect
            tmp = []
            for d in fin:
                if isinstance(d, dict):
                    tmp.append(d)
                else:
                    tmp.extend(d)
            fin = tmp
            # NOTE: single thread for packing
            print(f"Raw Dataset has {len(fin)} samples")
            fin = open(input_file_name, 'r', encoding='utf-8')
            pool = multiprocessing.Pool(self.args.workers, initializer=encoder.initializer, )
            encoded_docs_iter = pool.imap_unordered(self.process_doc, (fin, encoder), chunksize=25)
            #encoded_docs_iter = (encoder.encode(fin),)
        else:
            if self.args.debug:
                encoder.initializer()
                encoded_docs_iter = (encoder.encode_blocked(doc) for doc in fin)
            else:
                pool = multiprocessing.Pool(self.workers, initializer=encoder.initializer)
                encoded_docs_iter = pool.imap(encoder.encode_blocked, fin, 32)
        
        # 单轮数据切分出来20% 用于构造超长假多轮数据，剩余的
        encoded_rs_concat = []

        for one in tqdm(encoded_docs_iter):
            encoded_rs_concat.append(one)
        
        random.shuffle(encoded_rs_concat)
        split_list = [encoded_rs_concat[i: i + self.args.split_size] for i in range(0, len(encoded_rs_concat), self.args.split_size)]
        split_iter = iter(split_list)
        print(f'  > concat split into {len(split_list)} chunk')
        if self.args.workers > 1:
            concat_iter = pool.imap_unordered(encoder.concat, split_iter, chunksize=100)
        else:
            concat_iter = []
            for iter_ in tqdm(split_iter, desc='Concating'):
                concat_iter.append(encoder.concat(iter_))
        
        rs = []
        for one in concat_iter:
            rs.extend(one)
        
        encoded_docs = encoder.concat_sample_idx(rs, 'tmp.txt')

        # tokenizer = build_tokenizer(self.args)
        level = "document"
        output_bin_files = {}
        output_idx_files = {}
        builders = {}
        for key in self.args.json_keys:
            output_bin_files[key] = "{}_{}_{}.bin".format(output_prefix,
                                                          key, level)
            output_idx_files[key] = "{}_{}_{}.idx".format(output_prefix,
                                                          key, level)
            builders[key] = indexed_dataset.IndexedDatasetBuilder(
                output_bin_files[key],
                dtype=indexed_dataset.DType.optimal_dtype(encoder.tokenizer.vocab_size),
            )

        startup_end = time.time()
        proc_start = time.time()
        total_bytes_processed = 0
        print("Time to startup:", startup_end - startup_start)
        cnt = 1
        for datas in encoded_docs:
            for (doc, sentence_lens, bytes_processed) in datas:
                total_bytes_processed += bytes_processed
                for key in doc.keys():
                    builders[key].add_document(doc[key], sentence_lens[key])
                self.print_processing_stats(cnt, proc_start, total_bytes_processed)
                cnt += 1
        print(f"After pre-tokenizing, the idxmap dataset has {cnt - 1} samples")

        builders[key].finalize(output_idx_files[key])

def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='input data')
    group.add_argument('--input', type=str, required=True,
                       help='Path to input JSON')
    group.add_argument('--json-keys', nargs='+', default=['text'],
                       help='space separate listed of keys to extract from json')
    group.add_argument('--keep-newlines', action='store_true',
                       help='Keep newlines between sentences when splitting.')

    group = parser.add_argument_group(title='tokenizer')
    group.add_argument('--tokenizer-type', type=str, required=False, default='GPT2BPETokenizer',
                       choices=['BertWordPieceLowerCase','BertWordPieceCase',
                                'GPT2BPETokenizer', 'SentencePieceTokenizer',
                                'GPTSentencePieceTokenizer', 'Llama2Tokenizer',
                                'NullTokenizer'],
                       help='What type of tokenizer to use.')
    group.add_argument('--sequence-packing',action='store_true', help='packing sequence')
    group.add_argument('--tokenizer-model', type=str, default=None,
                       help='YTTM tokenizer model.')
    group.add_argument('--vocab-file', type=str, default=None,
                       help='Path to the vocab file')
    group.add_argument('--vocab-size', default=786,
                       help='size of vocab for use with NullTokenizer')
    group.add_argument('--merge-file', type=str, default=None,
                       help='Path to the BPE merge file (if necessary).')
    group.add_argument('--append-eod', action='store_true',
                       help='Append an <eod> token to the end of a document.')
    group.add_argument('--debug', action='store_true',
                       help='Append an <eod> token to the end of a document.')
    group.add_argument('--lang', type=str, default='english',
                       help='Language to use for NLTK-powered sentence splitting.')
    group = parser.add_argument_group(title='output data')
    group.add_argument('--output-prefix', type=str, required=True,
                       help='Path to binary output file without suffix')

    group = parser.add_argument_group(title='runtime')
    group.add_argument('--workers', type=int, required=True,
                       help=('Number of worker processes to launch.'
                             'A good default for fast pre-processing '
                             'is: (workers * partitions) = available CPU cores.'))
    group.add_argument('--partitions', type=int, default=1,
                        help='Number of file partitions')
    group.add_argument('--log-interval', type=int, default=1000,
                       help='Interval between progress updates')
    group.add_argument('--keep-sequential-samples', action='store_true',
                       help='Ensure ordering of samples in .jsonl files is '
                            'preserved when using partitions>1.')
    group.add_argument(
        '--patch-tokenizer-type',
        type=str,
        required=True,
        choices=['Qwen2Tokenizer', 'LLamaTokenizer', 'DeepSeekV2Tokenizer', 'LLama3Tokenizer'],
        help='What type of tokenizer to use.',
    )
    group.add_argument('--load',
                       type=str,
                       default=None,
                       help='path to tokenizer config file')

    group.add_argument('--seq-length',
                       type=int,
                       default=2048,
                       help='sequence length')

    group.add_argument('--extra-vocab-size',
                       type=int,
                       default=0,
                       help='extra_vocab_size')
    group.add_argument(
        "--split_size", type=int, default=1, help="Number of worker processes to launch"
    )

    args = parser.parse_args()
    args.keep_empty = False

    if args.tokenizer_type.lower().startswith('bert') and not args.split_sentences:
        print("Are you sure you don't want to split sentences?")
    
    if args.sequence_packing:
        print('Use internal single-threaded sequence packing..')
    # some default/dummy values for the tokenizer
    args.rank = 1
    args.make_vocab_size_divisible_by = 128
    args.tensor_model_parallel_size = 1
    args.vocab_extra_ids = 0

    return args


def get_file_name(args, file_id):
    file_name, extension = os.path.splitext(args.input)
    input_file_name = file_name + "_" + str(file_id) + extension
    sentence_split_file = file_name + "_ss_" + str(file_id) + extension
    output_prefix = args.output_prefix + "_" + str(file_id)
    file_names = {
        'partition': input_file_name,
        'sentence_split': sentence_split_file,
        'output_prefix': output_prefix}
    return file_names


def check_files_exist(in_ss_out_names, key, num_partitions):
    for i in range(num_partitions):
        if not os.path.exists(in_ss_out_names[i][key]):
            return False
    return True

def print_processing_stats(count, proc_start, total_bytes_processed, args):
    if count % args.log_interval == 0:
        current = time.time()
        elapsed = current - proc_start
        mbs = total_bytes_processed/elapsed/1024/1024
        print(f"Processed {count} documents",
                f"({count/elapsed} docs/s, {mbs} MB/s).",
                file=sys.stderr)

def process_doc(doc, encoder):
    """
    处理单个文档的函数，调用 encoder.encode 并返回结果列表
    """
    return list(encoder.encode(doc))
def main():
    args = get_args()
    print(f'args is:{args}', flush=True)

    assert args.workers % args.partitions == 0
    
    startup_start = time.time()
    encoder = Encoder(args)
    if not args.sequence_packing:
        return
     
    fin = open(args.input, 'r', encoding='utf-8')
    lines = []
    for line in fin:
        lines.append(line)
    print(f'input data path is:{args.input}', flush=True)
    if args.workers > 1:
        pool = multiprocessing.Pool(args.workers, initializer=encoder.initializer, )
        process_doc_with_encoder = partial(process_doc, encoder=encoder)
        encoded_docs_iter = pool.imap_unordered(process_doc_with_encoder, lines, chunksize=25)
        # encoded_docs_iter = pool.imap_unordered(process_doc, [(line, encoder) for line in lines], chunksize=25)
    else:
        encoder.initializer()
        encoded_docs_iter = []
        # lines = fin.readlines()
        random.shuffle(lines)
        for doc in tqdm(lines, desc='Tokenizing'):
            #print(f'doc is: {doc}', flush=True)
            #for one in encoder.encode(doc):
            #    encoded_iter.append(one)
            encoded_docs_iter.extend(list(encoder.encode(doc)))
  
    
    # 单轮数据切分出来20% 用于构造超长假多轮数据，剩余的
    encoded_rs_concat = []

    for one in tqdm(encoded_docs_iter):
        encoded_rs_concat.append(one)
    
    random.shuffle(encoded_rs_concat)
    split_list = [encoded_rs_concat[i: i + args.split_size] for i in range(0, len(encoded_rs_concat), args.split_size)]
    split_iter = iter(split_list)
    print(f'  > concat split into {len(split_list)} chunk')
    if args.workers > 1:
        concat_iter = pool.imap_unordered(encoder.concat, split_iter, chunksize=100)
    else:
        concat_iter = []
        for iter_ in tqdm(split_iter, desc='Concating'):
            concat_iter.append(encoder.concat(iter_))
    
    rs = []
    for one in concat_iter:
        rs.extend(one)
    
    # 保存文件内容中的数据
    parent_path = os.path.dirname(args.output_prefix)
    count_file_path= os.path.join(parent_path, 'count.txt')
    encoded_docs = encoder.concat_sample_idx(rs, count_file_path)

    # tokenizer = build_tokenizer(self.args)
    level = "document"
    output_bin_files = {}
    output_idx_files = {}
    builders = {}
    for key in args.json_keys:
        output_bin_files[key] = "{}_{}_{}.bin".format(args.output_prefix,
                                                        key, level)
        output_idx_files[key] = "{}_{}_{}.idx".format(args.output_prefix,
                                                        key, level)
        builders[key] = indexed_dataset.IndexedDatasetBuilder(
            output_bin_files[key],
            dtype=indexed_dataset.DType.optimal_dtype(encoder.tokenizer.vocab_size),
        )

    startup_end = time.time()
    proc_start = time.time()
    total_bytes_processed = 0
    print("Time to startup:", startup_end - startup_start)
    cnt = 1
    for datas in encoded_docs:
        #for (doc, sentence_lens, bytes_processed) in datas:
        doc, sentence_lens, bytes_processed = datas
        total_bytes_processed += bytes_processed
        for key in doc.keys():
            builders[key].add_document(doc[key], sentence_lens[key])
        print_processing_stats(cnt, proc_start, total_bytes_processed, args)
        cnt += 1
    print(f"After pre-tokenizing, the idxmap dataset has {cnt - 1} samples")

    builders[key].finalize(output_idx_files[key])



if __name__ == '__main__':
    main()