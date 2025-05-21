# Copyright (c) 2023 Alibaba PAI Team.  All rights reserved.
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
"""Processing data for pretraining."""

import argparse
import multiprocessing
import os
import sys
import time
from threading import Semaphore
import torch
import ftfy
import lm_dataformat as lmd
import tqdm
import json
import random 
from pathlib import Path
import traceback
from file import list_dirs_by_tag, list_files_by_tag

from megatron.core.datasets import indexed_dataset
from megatron_patch.tokenizer import build_tokenizer

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

LOG_DATA_NUM = 2

class Encoder(object):
    def __init__(self, args):
        self.args = args

    def initializer(self):
        # Use Encoder class as a container for global data
        Encoder.tokenizer = build_tokenizer(self.args)
        Encoder.tokenizer.begin_audio = Encoder.tokenizer('<|beginofaudio|>', add_special_tokens=False, padding='do_not_pad')['input_ids'][0] # <|beginofaudio|>
        Encoder.tokenizer.end_audio = Encoder.tokenizer('<|endofaudio|>', add_special_tokens=False, padding='do_not_pad')['input_ids'][0]  # <|endofaudio|>
        Encoder.tokenizer.padding = Encoder.tokenizer('<|endoftext|>', add_special_tokens=False, padding='do_not_pad')['input_ids'][0]  # <|endoftext|> 
        Encoder.tokenizer.inter_begin_audio = Encoder.tokenizer('<|inter_audio_begin|>', add_special_tokens=False, padding='do_not_pad')['input_ids'][0]  # <|inter_audio_begin|>
        Encoder.tokenizer.inter_end_audio = Encoder.tokenizer('<|inter_audio_end|>', add_special_tokens=False, padding='do_not_pad')['input_ids'][0]  # <|inter_audio_end|>
        print(f'Encoder.tokenizer.begin_audio: {Encoder.tokenizer.begin_audio}', flush=True)
        print(f'Encoder.tokenizer.end_audio: {Encoder.tokenizer.end_audio}', flush=True)
        print(f'Encoder.tokenizer.vocab_size is:{Encoder.tokenizer.vocab_size}', flush=True)
        print(f'Encoder.tokenizer.inter_begin_audio: {Encoder.tokenizer.inter_begin_audio}', flush=True)
        print(f'Encoder.tokenizer.inter_end_audio is:{Encoder.tokenizer.inter_end_audio}', flush=True)
    
    def get_sample_ids(self, json_data, debug=True):
        """_summary_

        Args:
            json_data (_type_): _description_
        """
        curr_turn = 0
        
        sample_ids = []
        for dialog in json_data['dialog_list'] if 'dialog_list' in json_data else json_data['dialog_result']:
            data_type = dialog['type'] if 'type' in dialog else 'speech2speech'

            if data_type == 'asr':
                # step 1: 处理instruction
                text = dialog['prompt_text']
                text_ids = Encoder.tokenizer(text, add_special_tokens=False, padding='do_not_pad', max_length=32768,truncation=False)['input_ids']
                if debug:
                    print(f'text is:{text}', flush=True)
                    print(f'text_ids is:{text_ids}', flush=True)

                # step 2: 处理音频
                prompt_audio_path = dialog['prompt_audio_path']
                prompt_audio_path = prompt_audio_path.replace('.wav', self.args.postfix)
                if not os.path.isfile(prompt_audio_path):
                    print(f'skip data prompt_audio_path:{prompt_audio_path}')
                    return []
                
                if debug:
                    print(f'prompt_audio_path is:{prompt_audio_path}', flush=True)

                sample_ids.extend(text_ids)
                sample_ids.append(Encoder.tokenizer.inter_begin_audio)

                
                with open(prompt_audio_path, 'r', encoding='utf-8') as fin:
                    data = fin.readline()
                    data_dict = json.loads(data)
                    audio_ids = data_dict['audio_token']
                    sample_ids.extend(audio_ids)
                    # 前后需要插入音频的特殊token
                sample_ids.append(Encoder.tokenizer.inter_end_audio)

                # step 3: 处理输出文本
                output_text = dialog['output_text']
                output_text_ids = Encoder.tokenizer(output_text, add_special_tokens=False, padding='do_not_pad', max_length=32768,truncation=False)['input_ids']
                sample_ids.extend(output_text_ids)
                if debug:
                    print(f'output_text is:{output_text}', flush=True)
                    print(f'output_text_ids is:{output_text_ids}', flush=True)
                
            elif data_type == 'tts':
                # step 1: 处理instruction
                text = dialog['prompt_text']
                text_ids = Encoder.tokenizer(text, add_special_tokens=False, padding='do_not_pad', max_length=32768,truncation=False)['input_ids']
                
                output_audio_path = dialog['output_audio_path']
                output_audio_path = output_audio_path.replace('.wav', self.args.postfix)

                if not os.path.isfile(output_audio_path):
                    print(f'skip data output_audio_path:{output_audio_path}')
                    return []

                sample_ids.extend(text_ids)

                # step 2: 处理音频
                
                with open(output_audio_path, 'r', encoding='utf-8') as fin:
                    data = fin.readline()
                    data_dict = json.loads(data)
                    audio_ids = data_dict['audio_token']

                    # 前后需要插入音频的特殊token
                    sample_ids.append(Encoder.tokenizer.begin_audio)
                    sample_ids.extend(audio_ids)
                    sample_ids.append(Encoder.tokenizer.end_audio)

            elif data_type == 'audio':

                

                sample_ids.append(Encoder.tokenizer.begin_audio)
                with open(prompt_audio_path, 'r', encoding='utf-8') as fin:
                    data = fin.readline()
                    data_dict = json.loads(data)
                    audio_ids = data_dict['audio_token']
                    sample_ids.extend(audio_ids)
                    # 前后需要插入音频的特殊token
                sample_ids.append(Encoder.tokenizer.end_audio)
            
            elif data_type == 'speech2speech':
                if curr_turn == 0:
                    text = json_data['instruction'] + '先输出文本再输出音频。'
                    text_ids = Encoder.tokenizer(text, add_special_tokens=False, padding='do_not_pad', max_length=32768,truncation=False)['input_ids']
                    sample_ids.extend(text_ids)

                if curr_turn % 2 == 0:
                    role = json_data['role_A'] + ':'
                    role_ids = Encoder.tokenizer(role, add_special_tokens=False, padding='do_not_pad', max_length=32768,truncation=False)['input_ids']
                    sample_ids.extend(role_ids)
                else:

                    role = json_data['role_B'] + ':'
                
                    role_ids = Encoder.tokenizer(role, add_special_tokens=False, padding='do_not_pad', max_length=32768,truncation=False)['input_ids']
                    sample_ids.extend(role_ids)
                    # 音频对应的文本
                    text_ids = Encoder.tokenizer(dialog['text'], add_special_tokens=False, padding='do_not_pad', max_length=32768,truncation=False)['input_ids']
                    sample_ids.extend(text_ids)
                # step 2: 处理音频
                prompt_audio_path = dialog['audio_path']
                prompt_audio_path = prompt_audio_path.replace('.wav', self.args.postfix)
                if not os.path.isfile(prompt_audio_path):
                    print(f'skip data prompt_audio_path:{prompt_audio_path}')
                    return []
                
                if debug:
                    print(f'prompt_audio_path is:{prompt_audio_path}', flush=True)

                #sample_ids.extend(text_ids)
                sample_ids.append(Encoder.tokenizer.inter_begin_audio)

                
                with open(prompt_audio_path, 'r', encoding='utf-8') as fin:
                    data = fin.readline()
                    data_dict = json.loads(data)
                    audio_ids = data_dict['audio_token']
                    sample_ids.extend(audio_ids)
                    # 前后需要插入音频的特殊token
                sample_ids.append(Encoder.tokenizer.inter_end_audio)

                curr_turn += 1

                
            
        return sample_ids
    
    def find_txt_files(self, input_dir):
        txt_files = []
        filter_file_num = 0
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if file.endswith('.txt'):
                    audio_file = file.replace('.txt', self.args.postfix)
                    audio_file = os.path.join(root, audio_file)

                    if not os.path.isfile(audio_file):
                        filter_file_num += 1
                        continue
                    
                    txt_files.append(os.path.join(root, file))
        #print(f'lost audio file num is:{filter_file_num}', flush=True)
        return txt_files
    
    def find_file_in_dirs(self, input_dir):
        files = os.listdir(input_dir)

        # 筛选以 '.jsonl' 结尾的文件
        tag_files = [os.path.join(input_dir, file) for file in files if file.endswith(self.args.file_tag)]
        return tag_files
    
    def find_files_by_tag(self, input_dir):
        tag_files = []
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                
                if file.endswith(self.args.postfix):           
                    tag_files.append(os.path.join(root, file))
        return tag_files
    
    def load_file_data(self, file_path):
        data_list = []
        with open(file_path, 'r', encoding='utf-8') as fin:
            for line in fin:
                try:
                    json_data = json.loads(line)
                    data_list.append(json_data)
                except Exception as e:
                    print(f'exception json_data:{json_data}')
                    traceback.print_exc()
                    continue
        return data_list


    def encode_audio(self, text, debug=False):
        
        input_dir = json.loads(text)['path']
        ids = {}
        text_len = []
        #file_list = os.listdir(input_dir)
        #file_list = self.find_files_by_tag(input_dir)
        file_list = [input_dir]

        for key in self.args.jsonl_keys:
            
            doc_ids = [] # 添加上bos token
            doc_ids_list = []

            for file_path in file_list:
                json_data = self.load_file_data(file_path=file_path)[0]
                sample_ids = []
                try:
                    #print(f'file_path is:{file_path}', flush=True)
                    #print(f'json_data is:{json_data}', flush=True)
                    if json_data is None:
                        continue
                    audio_ids = json_data['audio_token']
                    if debug:
                        print(f'sample_ids is:{sample_ids}', flush=True)
                        import sys
                        sys.exit(1)

                    if len(audio_ids) == 0:
                        continue

                    sample_ids.append(Encoder.tokenizer.begin_audio)
                    sample_ids.extend(audio_ids)
                    sample_ids.append(Encoder.tokenizer.end_audio)

                    if self.args.append_eod:
                        if hasattr(Encoder.tokenizer, 'eos_token_id'):
                            sample_ids.append(Encoder.tokenizer.eos_token_id)
                        elif hasattr(Encoder.tokenizer, 'eod_id'):
                            sample_ids.append(Encoder.tokenizer.eod_id)
                        else:
                            sample_ids.append(Encoder.tokenizer.eod)
                        #doc_ids[-1].append(Encoder.tokenizer.pad_token_id)

                    if len(doc_ids) + len(sample_ids) > self.args.seq_length and len(doc_ids) > 0:
                        text_len.append(len(doc_ids))
                        doc_ids_list.append(doc_ids)
                        doc_ids = [] # 添加上bos token
                    doc_ids.extend(sample_ids)

                    global LOG_DATA_NUM
                    if self.args.log_data and LOG_DATA_NUM > 0:
                        LOG_DATA_NUM -= 1
                        deocde_content = Encoder.tokenizer.detokenize(sample_ids)
                        print(
                            "text before process: \n",
                            json_data,
                            "\nid after processed: \n",
                            sample_ids, 
                            "\ndeocde content\n",
                            deocde_content,
                            flush=True,
                        )
                    
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print(f"Error encoding text: {e}")  # print error message
                    continue
            if len(doc_ids) > 0:
                text_len.append(len(doc_ids))
                doc_ids_list.append(doc_ids)
                doc_ids = [] 
                
            ids[key] = doc_ids_list
            
        # 如果不做拼接，局部内需要做拼接
        if not self.args.concat:
            ids, text_len = self.concat([[ids, text_len]])
        return ids, text_len
    
    def encode_multi_modal(self, text, debug=False):
        input_dir = json.loads(text)['path']
        #print(text, flush=True)
        ids = {}
        text_len = []
        #file_list = os.listdir(input_dir)
        #file_list = self.find_file_in_dirs(input_dir)
        file_list = [input_dir]

        for key in self.args.jsonl_keys:
            doc_ids = [] # 添加上bos token
            doc_ids_list = []

            for file_path in file_list:
                data_list = self.load_file_data(file_path=file_path)
                for json_data in data_list:
                    try:
                        if json_data is None:
                            continue
                        sample_ids = self.get_sample_ids(json_data, debug)
                        # print(f'sample_ids is:{sample_ids}', flush=True)
                        if debug:
                            print(f'sample_ids is:{sample_ids}', flush=True)
                            import sys
                            sys.exit(1)
                        

                        if len(sample_ids) == 0:
                            continue

                        #if max(sample_ids) >= Encoder.tokenizer.vocab_size:
                        #    print(max(sample_ids))
                        #    continue

                        if self.args.append_eod:
                            if hasattr(Encoder.tokenizer, 'eos_token_id'):
                                sample_ids.append(Encoder.tokenizer.eos_token_id)
                            elif hasattr(Encoder.tokenizer, 'eod_id'):
                                sample_ids.append(Encoder.tokenizer.eod_id)
                            else:
                                sample_ids.append(Encoder.tokenizer.eod)
                            #doc_ids[-1].append(Encoder.tokenizer.pad_token_id)

                        if len(doc_ids) + len(sample_ids) > self.args.seq_length and len(doc_ids) > 0:
                            text_len.append(len(doc_ids))
                            doc_ids_list.append(doc_ids)
                            doc_ids = [] # 添加上bos token
                        doc_ids.extend(sample_ids)

                        global LOG_DATA_NUM
                        if self.args.log_data and LOG_DATA_NUM > 0:
                            LOG_DATA_NUM -= 1
                            deocde_content = Encoder.tokenizer.detokenize(doc_ids)
                            print(
                                "text before process: \n",
                                json_data,
                                "\nid after processed: \n",
                                doc_ids, 
                                "\ndeocde content\n",
                                deocde_content,
                                flush=True
                            )
                        
                    except Exception as e:
                        
                        traceback.print_exc()
                        print(f"Error encoding text: {e}")  # print error message
                        continue
                
            if len(doc_ids) > 0:
                text_len.append(len(doc_ids))
                doc_ids_list.append(doc_ids)
                doc_ids = []
            ids[key] = doc_ids_list
            
        # 如果不做拼接，局部内需要做拼接
        if not self.args.concat:
            try:
                ids, text_len = self.concat([[ids, text_len]])
            except Exception as e:
                traceback.print_exc()
                print(f'exception is:{e}', flush=True)
                return ids, text_len
       # print(f'ids:{ids}', flush=True)
        return ids, text_len
    
    def concat(self, encode_rs):
        """对上一步产生的enocde结果进行拼接

        Args:
            encode_rs (_type_): _description_

        Returns:
            _type_: _description_
        """
        sample_len_list = []
        sample_list = []
        sample_len = 0
        token_ids = []

        concat_ids_dict = {}

        for ids_dict, num_list in encode_rs:
            for key in self.args.jsonl_keys:
                ids_list = ids_dict[key] 
                for ids, num in zip(ids_list, num_list):
                
                    if len(token_ids) + len(ids) > self.args.seq_length and len(token_ids) != 0:

                        if self.args.padding_seq:
                            token_ids = token_ids + [Encoder.tokenizer.padding] * (self.args.seq_length - len(token_ids))
                            sample_len = self.args.seq_length
                        sample_list.append(token_ids)
                        sample_len_list.append(sample_len)

                        sample_len= 0 
                        token_ids = []
                    token_ids.extend(ids)
                    sample_len += num
                if len(token_ids) > 0:
                    sample_list.append(token_ids)
                    sample_len_list.append(sample_len)

                concat_ids_dict[key] = sample_list
        return concat_ids_dict, sample_len_list


    def encode(self, text):
        #print(f'text is :{text}', flush=True)
        #ext_json = json.loads(text)
        #text = text_json['text'] if 'text' in text_json else text_json['content']
        
        if self.args.ftfy:
            text = ftfy.fix_text(text)
        ids = {}
        text_len = []
        for key in self.args.jsonl_keys:
            doc_ids = []
            try:
                text_ids = Encoder.tokenizer(text, add_special_tokens=False, padding='do_not_pad')['input_ids']                
                """
                text_ids = Encoder.tokenizer(text, add_special_tokens=False, padding='max_length',
                                             max_length=2047, truncation=True)['input_ids']
                """
                if max(text_ids) >= Encoder.tokenizer.vocab_size:
                    print(text)
                    print(max(text_ids))
                    continue
            except Exception as e:
                
                traceback.print_exc()
                print(f"Error encoding text: {e}")  # print error message
                continue
            if len(text_ids) > 0:
                doc_ids.append(text_ids)
                text_len.append(len(text_ids))
            if self.args.append_eod:
                if hasattr(Encoder.tokenizer, 'eos_token_id'):
                    doc_ids[-1].append(Encoder.tokenizer.eos_token_id)
                elif hasattr(Encoder.tokenizer, 'eod_id'):
                    doc_ids[-1].append(Encoder.tokenizer.eod_id)
                else:
                    doc_ids[-1].append(Encoder.tokenizer.eod)
                #doc_ids[-1].append(Encoder.tokenizer.pad_token_id)
            ids[key] = doc_ids

            global LOG_DATA_NUM
            if self.args.log_data and LOG_DATA_NUM > 0:
                LOG_DATA_NUM -= 1
                print(
                    "text before process: \n",
                    text,
                    "\nid after processed: \n",
                    ids[key], flush=True
                )

        #return ids, len(text)

    
        return ids, text_len

def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='input data')
    group.add_argument('--input_list',  nargs='+', help='List of items')
    group.add_argument('--input', type=str)
    group.add_argument('--input_cache', type=str)
    group.add_argument('--file_tag', type=str, default='.jsonl')
    group.add_argument('--type', type=str, default='text')
    group.add_argument(
        '--jsonl-keys',
        nargs='+',
        default=['content'],
        help='space separate listed of keys to extract from jsonl. Defa',
    )
    group.add_argument(
        '--num-docs',
        default=None,
        type=int,
    )
    group = parser.add_argument_group(title='tokenizer')
    group.add_argument(
        '--patch-tokenizer-type',
        type=str,
        required=True,
        choices=[
            'JiebaBPETokenizer', 'BloomTokenizerFromHF',
            'ChatGLMTokenizerFromHF', 'GPT2BPETokenizer',
            'GLM10BZHTokenizerFromHF', 'IcetkGLM130BTokenizer',
            'LLamaTokenizer', 'FalconTokenizer', 'OPTTokenizer',
            'StarcoderTokenizerFromHF', 'QwenTokenizer','Qwen2Tokenizer', 'MistralTokenizer'
        ],
        help='What type of tokenizer to use.',
    )
    group.add_argument('--vocab-file',
                       type=str,
                       default=None,
                       help='Path to the vocab file')

    group.add_argument(
        '--merge-file',
        type=str,
        default=None,
        help='Path to the BPE merge file (if necessary).',
    )
    group.add_argument(
        '--append-eod',
        action='store_true',
        help='Append an <eod> token to the end of a document.',
    )
    group.add_argument(
        '--concat',
        action='store_true',
        help='concat sample to seqence length.',
    )
    group.add_argument(
        "--split_size", type=int, default=10000, help="Number of worker processes to launch"
    )
    group.add_argument(
        "--seq_length",
        type=int,
        default=8192,
    )
    group.add_argument(
        "--padding_seq",
        type=bool,
        default=False,
    )
    group.add_argument('--ftfy',
                       action='store_true',
                       help='Use ftfy to clean text')
    group = parser.add_argument_group(title='output data')
    group.add_argument(
        '--output-prefix',
        type=str,
        required=True,
        help='Path to binary output file without suffix',
    )
    group.add_argument(
        '--dataset-impl',
        type=str,
        default='mmap',
        choices=['lazy', 'cached', 'mmap'],
        help='Dataset implementation to use. Default: mmap',
    )

    group.add_argument("--postfix", type=str, default='.audio')

    group = parser.add_argument_group(title='runtime')
    group.add_argument('--workers',
                       type=int,
                       default=1,
                       help='Number of worker processes to launch')
    group.add_argument(
        '--log-interval',
        type=int,
        default=100,
        help='Interval between progress updates',
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
                       default=1,
                       help='extra_vocab_size')
    group.add_argument('--log_data',
                       type=bool,
                       default=True,
                       help='log data')
    args = parser.parse_args()
    args.keep_empty = False

    # some default/dummy values for the tokenizer
    args.rank = 0
    args.make_vocab_size_divisible_by = 128
    args.model_parallel_size = 1

    return args


def yield_from_files(fnames: list, semaphore, key="text"):
    def yielder(fname, semaphore):
        for f in filter(lambda x: x, lmd.Reader(fname).stream_data()):
            semaphore.acquire()
            '''
            try:
                data = json.loads(f)
                if key in data:
                    yield data[key]
            except Exception as e:
                continue
            '''
            
            yield f

    for fname in fnames:
        semaphore.acquire()

        yield from yielder(fname, semaphore)

def save_data(save_path, all_dirs_path):
    """_summary_

    Args:
        save_path (_type_): _description_
        all_dirs_path (_type_): _description_
    """

    with open(save_path, 'w', encoding='utf-8') as fout:
        for path in all_dirs_path:
            res = {}
            res['path'] = path
            fout.write(json.dumps(res, ensure_ascii=False) + '\n')

def prepare_cache_data(args):
    """遍历一个目录下的所有

    Args:
        args (_type_): _description_
    """
    
    all_dirs_path = []
    print(args.input_list)

    # 判断cache文件是否存在，如果不存在,首先生成cache文件，否则直接读取cache文件去处理数据
    if os.path.isfile(args.input_cache):
        return 

    for input_path in args.input_list:
        if not os.path.isdir(input_path):
            print(f'input_path dir not exit:{input_path}', flush=True)
        
        #dirs_path = list_dirs_by_tag(input_path, args.postfix)
        dirs_path = list_files_by_tag(input_path, args.file_tag)
        all_dirs_path.extend(dirs_path)
    
    #all_dirs_path = list(all_dirs_path)
    save_data(args.input_cache, all_dirs_path)

def prepare_audio_cache_data(args):
    """遍历一个目录下的所有

    Args:
        args (_type_): _description_
    """
    
    all_dirs_path = []
    print(args.input_list)

    # 判断cache文件是否存在，如果不存在,首先生成cache文件，否则直接读取cache文件去处理数据
    if os.path.isfile(args.input_cache):
        return 

    for input_path in args.input_list:
        if not os.path.isdir(input_path):
            print(f'input_path dir not exit:{input_path}', flush=True)
        
        #dirs_path = list_dirs_by_tag(input_path, args.postfix)
        dirs_path = list_files_by_tag(input_path, args.postfix)
        all_dirs_path.extend(dirs_path)
    
    all_dirs_path = list(set(all_dirs_path))
    save_data(args.input_cache, all_dirs_path)

def main():
    args = get_args()
    args.tensor_model_parallel_size = 1
    args.rank = 0
    args.make_vocab_size_divisible_by = 128
    args.vocab_extra_ids = 0
    print(f'args is :{args}', flush=True)
    encoder = Encoder(args)
    tokenizer = build_tokenizer(args)
    print(f'Vocab size: {tokenizer.vocab_size}')
    print(f'Output prefix: {args.output_prefix}')

    semaphore = Semaphore(10000 + args.workers)

    # use multiprocessing to iterate over input documents

    if args.type == 'text':
        # 纯文本模态支持多文件输入
        print(f' start text modal ...', flush=True)
        file_list = os.listdir(args.input)
        path_list = [os.path.join(args.input, file) for file in file_list]
        fin = yield_from_files(path_list, semaphore)

    elif args.type == 'audio':
         # 语音模态只支持单文件列表的输入
        print(f' start multi modal ...', flush=True)
        prepare_audio_cache_data(args=args)
        fin = open(args.input_cache, 'r', encoding='utf-8')

    else:
        # 语音模态只支持单文件列表的输入
        print(f' start multi modal ...', flush=True)
        prepare_cache_data(args=args)
        fin = open(args.input_cache, 'r', encoding='utf-8')


    # 判断处理的是纯文本数据还是多模态数据 还是音频token，使用不同tokenizer
    if args.type == 'multi_modal':
        encode_func = encoder.encode_multi_modal 
    elif args.type == 'audio':
        encode_func = encoder.encode_audio
    else:
        encode_func = encoder.encode

    if args.workers > 1:
        pool = multiprocessing.Pool(args.workers,
                                    initializer=encoder.initializer)
        encoded_docs = pool.imap(encode_func, fin, chunksize=25)
    else:
        encoder.initializer()
        encoded_docs = (encode_func(doc) for doc in fin)

    output_bin_files = {}
    output_idx_files = {}
    builders = {}
    for key in args.jsonl_keys:
        output_bin_files[key] = '{}_{}_{}.bin'.format(args.output_prefix, key,
                                                      'document')
        output_idx_files[key] = '{}_{}_{}.idx'.format(args.output_prefix, key,
                                                      'document')
        builders[key] = indexed_dataset.IndexedDatasetBuilder(
            output_bin_files[key],
            dtype=indexed_dataset.DType.optimal_dtype(tokenizer.vocab_size),
        )
    
    if args.concat:
        encoded_rs_concat = []
        for one in encoded_docs:
            encoded_rs_concat.append(one)
        random.shuffle(encoded_rs_concat)
        # actually do tokenization
        proc_start = time.time()
        total_bytes_processed = 0
        total_tokens_processed = 0
        pbar = tqdm.tqdm()

        split_list = [encoded_rs_concat[i: i + args.split_size] for i in range(0, len(encoded_rs_concat), args.split_size)]
        split_iter = iter(split_list)
        print(f'  > concat split into {len(split_list)} chunk')
        if args.workers > 1:
            concat_iter = pool.imap_unordered(encoder.concat, split_iter, chunksize=100)
        else:
            concat_iter = (encoder.concat(iter_) for iter_ in split_iter)
   
        for i, (doc, bytes_processed) in enumerate(concat_iter, start=1):

            total_bytes_processed += sum(bytes_processed) if isinstance(bytes_processed, list) else bytes_processed

            semaphore.release()

            # add each tokenized document / sentence
            for key, sentences in doc.items():
                for sentence in sentences:
                    #print(f'len(sentence) is:{len(sentence)}', flush=True)
                    total_tokens_processed += len(sentence)
                    
                    builders[key].add_item(torch.IntTensor(sentence))
                    # separate with eos token
                    builders[key].end_document()

            # log progress
            if i % args.log_interval == 0:
                current = time.time()
                elapsed = current - proc_start
                mbs = total_bytes_processed / elapsed / 1024 / 1024
                pbar.set_description(f'Processed {i} documents '
                                    f' ({i / elapsed} docs/s, {mbs} MB/s), number of tokens:{total_tokens_processed}.')
                if i != 0:
                    pbar.update(args.log_interval)
            
    else:
    
        # actually do tokenization
        proc_start = time.time()
        total_bytes_processed = 0
        total_tokens_processed = 0
        pbar = tqdm.tqdm()
        for i, (doc, bytes_processed) in enumerate(encoded_docs, start=1):
            total_bytes_processed += sum(bytes_processed) if isinstance(bytes_processed, list) else bytes_processed

            semaphore.release()

            # add each tokenized document / sentence
            for key, sentences in doc.items():
                for sentence in sentences:
                    total_tokens_processed += len(sentence)
                    print(f'total_tokens_processed is:{total_tokens_processed}', flush=True)
                    builders[key].add_item(torch.IntTensor(sentence))
                    # separate with eos token
                    builders[key].end_document()

            # log progress
            if i % args.log_interval == 0:
                current = time.time()
                elapsed = current - proc_start
                mbs = total_bytes_processed / elapsed / 1024 / 1024
                pbar.set_description(f'Processed {i} documents '
                                    f' ({i / elapsed} docs/s, {mbs} MB/s), number of tokens:{total_tokens_processed}.')
                if i != 0:
                    pbar.update(args.log_interval)

    # save output file
    for key in args.jsonl_keys:
        builders[key].finalize(output_idx_files[key])
    
    # 保存这批数据一共产生了多少token
    with open(args.output_prefix + '-token_count.txt', 'w', encoding='utf-8') as fout:
        fout.write(f'total tokens is:{total_tokens_processed}')


if __name__ == '__main__':
    main()