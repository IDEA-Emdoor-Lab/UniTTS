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

import torch
from megatron.core import mpu
try:
    from megatron import get_args
except:
    from megatron.training import get_args

from megatron_patch.tokenizer import get_tokenizer
from megatron.training import print_rank_0
import time

SHOW_DATA=True

def process_sample(data_dict, max_padding_length, tokenizer, IGNORE_INDEX):
    # 使用 PyTorch 创建 position_ids 和 attention_mask
    position_ids = torch.arange(max_padding_length, dtype=torch.int64)
    attention_mask = torch.zeros((max_padding_length, max_padding_length), dtype=torch.int64)

    # 处理 sample_idx 生成 attention_mask 和 position_ids
    for sidx in data_dict["sample_idx"]:
        start, end = sidx[0], sidx[1]
        length = end - start

        # 生成下三角矩阵作为 attention_mask
        attention_mask[start:end, start:end] = torch.tril(
            torch.ones((length, length), dtype=torch.int64)
        )[:max_padding_length, :max_padding_length]

        # 生成 position_ids
        position_ids[start:end] = torch.arange(length, dtype=torch.int64)[:max_padding_length]

    # 处理 text 和 loss_mask
    data_dict["text"] = data_dict["text"] + [tokenizer.pad_token_id] * max_padding_length
    data_dict["loss_mask"] = data_dict["loss_mask"] + [IGNORE_INDEX] * max_padding_length
    
    # 防止截断后 loss_mask 全为 0
    if sum(data_dict["loss_mask"][:max_padding_length - 1]) == 0:
        data_dict["loss_mask"][max_padding_length - 2] = 1

    # 构建样本
    sample = {
        "input_ids": torch.tensor(
            data_dict["text"][:max_padding_length] + [tokenizer.pad_token_id], dtype=torch.int64
        ),
        "loss_mask": torch.tensor(
            data_dict["loss_mask"][:max_padding_length], dtype=torch.int64
        ),
        "attention_mask": attention_mask[:max_padding_length, :max_padding_length],
        "position_ids": position_ids[:max_padding_length],
    }
    return sample

def get_ltor_masks_and_position_ids_sft(data,
                                    eod_token,
                                    reset_position_ids,
                                    reset_attention_mask,
                                    eod_mask_loss,
                                    args,
                                    tokenizer):
    """_summary_

    Args:
        data (_type_): _description_
        eod_token (_type_): _description_
        reset_position_ids (_type_): _description_
        reset_attention_mask (_type_): _description_
        eod_mask_loss (_type_): _description_
        create_attention_mask (bool, optional): _description_. Defaults to True.
    """
    attention_mask, loss_mask, position_ids = process_sample(data, args.max_padding_length, tokenizer, tokenizer.pad_token_id)

    return attention_mask, loss_mask, position_ids
def get_ltor_masks_and_position_ids(data,
                                    eod_token,
                                    reset_position_ids,
                                    reset_attention_mask,
                                    eod_mask_loss,
                                    create_attention_mask: bool=True):
    """Build masks and position id for left to right model."""

    # Extract batch size and sequence length.
    micro_batch_size, seq_length = data.size()

    # Attention mask (lower triangular).
    if reset_attention_mask:
        att_mask_batch = micro_batch_size
    else:
        att_mask_batch = 1
    if create_attention_mask:
        attention_mask = torch.tril(torch.ones(
            (att_mask_batch, seq_length, seq_length), device=data.device)).view(
                att_mask_batch, 1, seq_length, seq_length)
    else:
        attention_mask = None

    # Loss mask.
    loss_mask = torch.ones(data.size(), dtype=torch.float, device=data.device)
    if eod_mask_loss:
        loss_mask[data == eod_token] = 0.0

    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long,
                                device=data.device)
    position_ids = position_ids.unsqueeze(0).expand_as(data)
    # We need to clone as the ids will be modifed based on batch index.
    if reset_position_ids:
        position_ids = position_ids.clone()

    if reset_position_ids or reset_attention_mask:
        # Loop through the batches:
        for b in range(micro_batch_size):

            # Find indecies where EOD token is.
            eod_index = position_ids[b, data[b] == eod_token]
            # Detach indecies from positions if going to modify positions.
            if reset_position_ids:
                eod_index = eod_index.clone()

            # Loop through EOD indecies:
            prev_index = 0
            for j in range(eod_index.size()[0]):
                i = eod_index[j]
                # Mask attention loss.
                if reset_attention_mask and attention_mask is not None:
                    attention_mask[b, 0, (i + 1):, :(i + 1)] = 0
                # Reset positions.
                if reset_position_ids:
                    position_ids[b, (i + 1):] -= (i + 1 - prev_index)
                    prev_index = i + 1

    if attention_mask is not None:
        # Convert attention mask to binary:
        attention_mask = (attention_mask < 0.5)

    return attention_mask, loss_mask, position_ids

def get_batch_on_this_tp_rank_original(data_iterator, per_seq_average=False):
    args = get_args()
    tokenizer = get_tokenizer()
    def _broadcast(item):
        if item is None:
            return
        torch.distributed.broadcast(item, mpu.get_tensor_model_parallel_src_rank(),
                                    group=mpu.get_tensor_model_parallel_group())

    if mpu.get_tensor_model_parallel_rank() == 0:

        if isinstance(data_iterator, dict):
            data = data_iterator
        else:
            data = next(data_iterator)

       
        # core/tensor_parallel/cross_entropy.py, target_mask = (target < vocab_start_index) | (target >= vocab_end_index)
        # labels[labels == tokenizer.eos_token_id] = -100
        #labels[labels == tokenizer.pad_token_id] = -100
        if args.train_mode == 'finetune':
            '''
            tokens_ = torch.tensor(data['input_ids'], dtype=torch.int64).long()
            tokens_ = tokens_ + torch.tensor([tokenizer.eos_token_id])
            tokens = tokens_[:, :-1].contiguous()
            labels = tokens_[:, 1:].contiguous()
            attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids_sft(
                data,
                -100,
                args.reset_position_ids,
                args.reset_attention_mask,
                args.eod_mask_loss,
                args,
                tokenizer
                )
            '''
            # print(f'data["loss_mask"] is:{data["loss_mask"][0]}', flush=True)
            tokens_ = data['input_ids'].long()
            #bs = tokens_.size(0)
            #new_element = torch.tensor([tokenizer.eos_token_id] * bs).reshape(bs, 1)  # 新元素的形状为 (M, 1)

            # 使用 torch.cat 在最后一维拼接
            #tokens_ = torch.cat((tokens_, new_element), dim=1)

            labels = tokens_[:, 1:].contiguous()
            tokens = tokens_[:, :-1].contiguous()

            loss_mask = data["loss_mask"].float()
            attention_mask = data["attention_mask"].unsqueeze(dim=1)
            attention_mask = attention_mask < 0.5
            position_ids = data["position_ids"].long()


        else:
            tokens_ = data['input_ids'].long()
            labels_ = data['labels'].long()
            attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
                labels,
                -100,
                args.reset_position_ids,
                args.reset_attention_mask,
                args.eod_mask_loss)
        
        
        
        # 判断是否为0卡，如果是0卡，则直接输出第一条样本验证是否有问题
        # print the sample at start
        global SHOW_DATA
        if SHOW_DATA:
            SHOW_DATA = False
            print_rank_0("tokens: {}, shape: {}".format(tokens[0], tokens.shape))
            print_rank_0("labels: {}, shape: {}".format(labels[0], labels.shape))
            print_rank_0(
                "attention_mask: {}, shape: {}".format(
                    attention_mask[0], attention_mask.shape
                )
            )
            print_rank_0("loss_mask: {}".format(loss_mask[0]))
            print_rank_0("position_ids: {}".format(position_ids[0]))

            print_rank_0("tokens text: {}".format(tokenizer.detokenize(tokens[0].tolist())))
            print_rank_0(
                "labels text : {}".format(tokenizer.detokenize(labels[0].tolist()))
            )
            for t in range(labels.shape[1]):
                mask = attention_mask[0][0][t]
                mask = torch.sum(~mask)
                print_rank_0(
                    "position: {}, loss_mask: {}, attention_mask: {}, token: {}, label: {}".format(
                        position_ids[0, t], loss_mask[0, t], mask, tokens[0, t], labels[0, t]
                    )
                )
        
        
        num_seqs = None
        if per_seq_average:
            # NOTE: raw dataset does not support sequence packing
            num_seqs = loss_mask.sum(dim=-1).long() # [mbs]
            loss_mask = loss_mask / num_seqs.view(-1, 1)

        batch = {
            'tokens': tokens.cuda(non_blocking=True),
            'labels': labels.cuda(non_blocking=True),
            'loss_mask': loss_mask.cuda(non_blocking=True),
            'attention_mask': attention_mask.cuda(non_blocking=True),
            'position_ids': position_ids.cuda(non_blocking=True),
            'num_seqs': num_seqs.cuda(non_blocking=True) if num_seqs is not None else None
        }

        if args.pipeline_model_parallel_size == 1:
            _broadcast(batch['tokens'])
            _broadcast(batch['labels'])
            _broadcast(batch['loss_mask'])
            _broadcast(batch['attention_mask'])
            _broadcast(batch['position_ids'])
            _broadcast(batch['num_seqs'])

        elif mpu.is_pipeline_first_stage():
            _broadcast(batch['tokens'])
            _broadcast(batch['attention_mask'])
            _broadcast(batch['position_ids'])

        elif mpu.is_pipeline_last_stage():
            _broadcast(batch['labels'])
            _broadcast(batch['loss_mask'])
            _broadcast(batch['attention_mask'])
            _broadcast(batch['num_seqs'])

    else:

        tokens = torch.empty((args.micro_batch_size, args.seq_length), dtype=torch.int64,
                             device=torch.cuda.current_device())
        labels = torch.empty((args.micro_batch_size, args.seq_length), dtype=torch.int64,
                             device=torch.cuda.current_device())
        loss_mask = torch.empty((args.micro_batch_size, args.seq_length), dtype=torch.float32,
                                device=torch.cuda.current_device())
        mbs = args.micro_batch_size if args.reset_attention_mask else 1
        attention_mask = torch.empty((mbs, 1, args.seq_length, args.seq_length), dtype=torch.bool,
                                     device=torch.cuda.current_device())
        position_ids = torch.empty((args.micro_batch_size, args.seq_length), dtype=torch.int64,
                                   device=torch.cuda.current_device())
        
        num_seqs = None
        if per_seq_average:
            num_seqs = torch.empty((args.micro_batch_size,), dtype=torch.int64,
                                    device=torch.cuda.current_device()) 

        if args.pipeline_model_parallel_size == 1:
            _broadcast(tokens)
            _broadcast(labels)
            _broadcast(loss_mask)
            _broadcast(attention_mask)
            _broadcast(position_ids)
            _broadcast(num_seqs)

        elif mpu.is_pipeline_first_stage():
            labels = None
            loss_mask = None
            num_seqs = None

            _broadcast(tokens)
            _broadcast(attention_mask)
            _broadcast(position_ids)

        elif mpu.is_pipeline_last_stage():
            tokens = None
            position_ids = None

            _broadcast(labels)
            _broadcast(loss_mask)
            _broadcast(attention_mask)
            _broadcast(num_seqs)

        batch = {
            'tokens': tokens,
            'labels': labels,
            'loss_mask': loss_mask,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'num_seqs': num_seqs
        }

    return batch

def get_batch_on_this_tp_rank_idxmap_sft(data_iterator, per_seq_average=False):
    args = get_args()
    tokenizer = get_tokenizer()
    def _broadcast(item):
        if item is None:
            return
        torch.distributed.broadcast(item, mpu.get_tensor_model_parallel_src_rank(),
                                    group=mpu.get_tensor_model_parallel_group())
        
    if mpu.get_tensor_model_parallel_rank() == 0:

        if isinstance(data_iterator, dict):
            data = data_iterator
        else:
            data = next(data_iterator)

        # sanity check
        assert data['tokens'].shape[-1] == 2 * args.seq_length + 100
        actual_seqlen = args.seq_length
        data['tokens'] = data['tokens'].long()
        tokens = data['tokens'][..., :actual_seqlen]
        labels = data['tokens'][..., actual_seqlen:actual_seqlen*2]
        loss_mask = (labels != -100).float()
        
        attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
            tokens,
            tokenizer.eod,
            args.reset_position_ids,
            args.reset_attention_mask,
            False,
            args.create_attention_mask_in_dataloader
        )

        num_seqs = None
        if per_seq_average:
            num_seqs = torch.zeros(position_ids.shape[0], device=torch.cuda.current_device(), dtype=torch.int64)
            for b in range(position_ids.shape[0]):
                p = position_ids[b]
                start_indices = (p == 0).nonzero(as_tuple=True)[0]
                seqlens = start_indices[1:] - start_indices[:-1]
                num_seqs[b] = len(seqlens)
                seqlens = seqlens.cpu().numpy().tolist() + [p.shape[0] - start_indices[-1].item()]
                subseqs = torch.split(loss_mask[b], seqlens)    
                for start_idx, seqlen, subseq in zip(start_indices, seqlens, subseqs):
                    assert subseq.sum() > 0
                    loss_mask[b, start_idx: start_idx + seqlen] /= subseq.sum()      
                 

        # dtype: long, long, float, bool, long
        batch = {
            'tokens': tokens.cuda(non_blocking=True),
            'labels': labels.cuda(non_blocking=True),
            'loss_mask': loss_mask.cuda(non_blocking=True),
            'attention_mask': attention_mask.cuda(non_blocking=True) if attention_mask is not None else None,
            'position_ids': position_ids.cuda(non_blocking=True),
            'num_seqs': num_seqs.cuda(non_blocking=True) if num_seqs is not None else None
        }

        if args.pipeline_model_parallel_size == 1:
            _broadcast(batch['tokens'])
            _broadcast(batch['labels'])
            _broadcast(batch['loss_mask'])
            _broadcast(batch['attention_mask'])
            _broadcast(batch['num_seqs'])

        elif mpu.is_pipeline_first_stage():
            _broadcast(batch['tokens'])
            _broadcast(batch['attention_mask'])

        elif mpu.is_pipeline_last_stage():
            _broadcast(batch['labels'])
            _broadcast(batch['loss_mask'])
            _broadcast(batch['attention_mask'])
            _broadcast(batch['num_seqs'])
        
        _broadcast(batch['position_ids'])

    else:
        # dtype: long, long, float, bool, long
        tokens = torch.empty((args.micro_batch_size, args.seq_length), dtype=torch.int64,
                             device=torch.cuda.current_device())
        labels = torch.empty((args.micro_batch_size, args.seq_length), dtype=torch.int64,
                             device=torch.cuda.current_device())
        loss_mask = torch.empty((args.micro_batch_size, args.seq_length), dtype=torch.float32,
                                device=torch.cuda.current_device())
        
        attention_mask = None
        if args.create_attention_mask_in_dataloader:
            mbs = args.micro_batch_size if args.reset_attention_mask else 1
            attention_mask = torch.empty((mbs, 1, args.seq_length, args.seq_length), dtype=torch.bool,
                                        device=torch.cuda.current_device())
        position_ids = torch.empty((args.micro_batch_size, args.seq_length), dtype=torch.int64,
                                   device=torch.cuda.current_device())

        num_seqs = None
        if per_seq_average:
            num_seqs = torch.empty((args.micro_batch_size,), dtype=torch.int64,
                                    device=torch.cuda.current_device()) 
            
        if args.pipeline_model_parallel_size == 1:
            _broadcast(tokens)
            _broadcast(labels)
            _broadcast(loss_mask)
            _broadcast(attention_mask)
            _broadcast(num_seqs)

        elif mpu.is_pipeline_first_stage():
            labels = None
            loss_mask = None
            num_seqs = None

            _broadcast(tokens)
            _broadcast(attention_mask)

        elif mpu.is_pipeline_last_stage():
            tokens = None

            _broadcast(labels)
            _broadcast(loss_mask)
            _broadcast(attention_mask)
            _broadcast(num_seqs)

        _broadcast(position_ids)
        batch = {
            'tokens': tokens,
            'labels': labels,
            'loss_mask': loss_mask,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'num_seqs': num_seqs
        }

    return batch

def get_attention_mask_loss_mask(bs_tokens, bs_sample_idx, args, tokenizer=None, debug=False):
    micro_batch_size, seq_length = bs_tokens.size()
    sample_list = []
    max_len = args.seq_length
    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long,
                                device=bs_tokens.device)
    position_ids = position_ids.unsqueeze(0).expand_as(bs_tokens)

    # attention mask

    attention_mask = torch.tril(torch.ones(
            (micro_batch_size, seq_length, seq_length), device=bs_tokens.device)).view(
                micro_batch_size, 1, seq_length, seq_length)
    
    for bs_index in range(micro_batch_size):

        sample_idx = bs_sample_idx[bs_index]
        indices =  torch.where(sample_idx == -100)[0]
        index = indices[0].item()
        sample_idx = sample_idx[:index]
        # print(f'sample_idx is:{sample_idx}', flush=True)
        
        # print(item, flush=True)
        # position_ids = torch.arange(max_len)
        # attention_mask = torch.zeros((max_len, max_len))

        for i in range(0, len(sample_idx), 2):
            pair = sample_idx[i:i+2] #.numpy()
            #print(f'pair is:{pair}', flush=True)
            #print(f'pair[1] is:{pair[1]}', flush=True)
            attention_mask[bs_index, 0, pair[1] + 1:, :pair[1] + 1] = 0
            position_ids[bs_index, pair[0] : pair[1]] = torch.arange(pair[1] - pair[0])[: max_len]

            # 打印一下结果
            if debug and bs_index == 0:
                print_rank_0("tokens text: {}".format(tokenizer.detokenize(bs_tokens[0].tolist()[pair[0] : pair[1]])))
                
    
    attention_mask = (attention_mask < 0.5)

    return attention_mask, position_ids

scaling_factor=1000000
def get_batch_on_this_tp_rank_idxmap_sft_sample_idx(data_iterator, per_seq_average=False):
    args = get_args()
    tokenizer = get_tokenizer()
    def _broadcast(item):
        if item is None:
            return
        torch.distributed.broadcast(item, mpu.get_tensor_model_parallel_src_rank(),
                                    group=mpu.get_tensor_model_parallel_group())
        
    if mpu.get_tensor_model_parallel_rank() == 0:

        if isinstance(data_iterator, dict):
            data = data_iterator
        else:
            data = next(data_iterator)
        # print(f'get_batch data from the file is: {data}', flush=True)

        # sanity check
        assert data['tokens'].shape[-1] == 2 * args.seq_length + 100
        # print(f"data['tokens'] shape is:{data['tokens'].shape}", flush=True)
        # start_time = time.time()
        actual_seqlen = args.seq_length
        sample_idx_len = 100
        data['tokens'] = data['tokens'].long()

        tokens_src = data['tokens'][..., :actual_seqlen]

        # 在最后一个维度上面添加上padding token这样移位之后，正好为seq的长度
        tokens_concat = torch.cat(
                [tokens_src, torch.full(data['tokens'].shape[:-1] + (1,), fill_value=tokenizer.pad_token_id, dtype=data['tokens'].dtype)], 
                dim=-1
            )
        
        tokens = tokens_concat[..., :-1].contiguous()

        labels = tokens_concat[..., 1:].contiguous()
        # loss_mask = data['tokens'][..., actual_seqlen:actual_seqlen*2].float()
        loss_mask = data['tokens'][..., actual_seqlen:actual_seqlen*2].float() / scaling_factor
        sample_idx_list =  data['tokens'][..., actual_seqlen*2:actual_seqlen*2+sample_idx_len]
        
        #  loss_mask = (labels != -100).float()
        attention_mask, position_ids = get_attention_mask_loss_mask(tokens, sample_idx_list, args, tokenizer)
        #end_time = time.time()
        #print(f'total time spend in make data:{end_time-start_time}', flush=True)

        num_seqs = None
        if per_seq_average:
            num_seqs = torch.zeros(position_ids.shape[0], device=torch.cuda.current_device(), dtype=torch.int64)
            for b in range(position_ids.shape[0]):
                p = position_ids[b]
                start_indices = (p == 0).nonzero(as_tuple=True)[0]
                seqlens = start_indices[1:] - start_indices[:-1]
                num_seqs[b] = len(seqlens)
                seqlens = seqlens.cpu().numpy().tolist() + [p.shape[0] - start_indices[-1].item()]
                subseqs = torch.split(loss_mask[b], seqlens)    
                for start_idx, seqlen, subseq in zip(start_indices, seqlens, subseqs):
                    assert subseq.sum() > 0
                    loss_mask[b, start_idx: start_idx + seqlen] /= subseq.sum()   

        # 判断是否为0卡，如果是0卡，则直接输出第一条样本验证是否有问题
        # print the sample at start
        global SHOW_DATA
        if SHOW_DATA:
            SHOW_DATA = False
            print_rank_0("tokens: {}, shape: {}".format(tokens[0], tokens.shape))
            print_rank_0("labels: {}, shape: {}".format(labels[0], labels.shape))
            print_rank_0(
                "attention_mask: {}, shape: {}".format(
                    attention_mask[0], attention_mask.shape
                )
            )
            print_rank_0("loss_mask: {}".format(loss_mask[0]))
            print_rank_0("position_ids: {}".format(position_ids[0]))

            print_rank_0("tokens text: {}".format(tokenizer.detokenize(tokens[0].tolist())))
            print_rank_0(
                "labels text : {}".format(tokenizer.detokenize(labels[0].tolist()))
            )
            for t in range(labels.shape[1]):
                mask = attention_mask[0][0][t]
                mask = torch.sum(~mask)
                print_rank_0(
                    "position: {}, loss_mask: {}, attention_mask: {}, token: {}, label: {}".format(
                        position_ids[0, t], loss_mask[0, t], mask, tokens[0, t], labels[0, t]
                    )
                )   
                 

        # dtype: long, long, float, bool, long
        batch = {
            'tokens': tokens.cuda(non_blocking=True),
            'labels': labels.cuda(non_blocking=True),
            'loss_mask': loss_mask.cuda(non_blocking=True),
            'attention_mask': attention_mask.cuda(non_blocking=True) if attention_mask is not None else None,
            'position_ids': position_ids.cuda(non_blocking=True),
            'num_seqs': num_seqs.cuda(non_blocking=True) if num_seqs is not None else None
        }

        if args.pipeline_model_parallel_size == 1:
            _broadcast(batch['tokens'])
            _broadcast(batch['labels'])
            _broadcast(batch['loss_mask'])
            _broadcast(batch['attention_mask'])
            _broadcast(batch['num_seqs'])

        elif mpu.is_pipeline_first_stage():
            _broadcast(batch['tokens'])
            _broadcast(batch['attention_mask'])

        elif mpu.is_pipeline_last_stage():
            _broadcast(batch['labels'])
            _broadcast(batch['loss_mask'])
            _broadcast(batch['attention_mask'])
            _broadcast(batch['num_seqs'])
        
        _broadcast(batch['position_ids'])

    else:
        # dtype: long, long, float, bool, long
        tokens = torch.empty((args.micro_batch_size, args.seq_length), dtype=torch.int64,
                             device=torch.cuda.current_device())
        labels = torch.empty((args.micro_batch_size, args.seq_length), dtype=torch.int64,
                             device=torch.cuda.current_device())
        loss_mask = torch.empty((args.micro_batch_size, args.seq_length), dtype=torch.float32,
                                device=torch.cuda.current_device())
        
        attention_mask = None
        if args.create_attention_mask_in_dataloader:
            mbs = args.micro_batch_size if args.reset_attention_mask else 1
            attention_mask = torch.empty((mbs, 1, args.seq_length, args.seq_length), dtype=torch.bool,
                                        device=torch.cuda.current_device())
        position_ids = torch.empty((args.micro_batch_size, args.seq_length), dtype=torch.int64,
                                   device=torch.cuda.current_device())

        num_seqs = None
        if per_seq_average:
            num_seqs = torch.empty((args.micro_batch_size,), dtype=torch.int64,
                                    device=torch.cuda.current_device()) 
            
        if args.pipeline_model_parallel_size == 1:
            _broadcast(tokens)
            _broadcast(labels)
            _broadcast(loss_mask)
            _broadcast(attention_mask)
            _broadcast(num_seqs)

        elif mpu.is_pipeline_first_stage():
            labels = None
            loss_mask = None
            num_seqs = None

            _broadcast(tokens)
            _broadcast(attention_mask)

        elif mpu.is_pipeline_last_stage():
            tokens = None

            _broadcast(labels)
            _broadcast(loss_mask)
            _broadcast(attention_mask)
            _broadcast(num_seqs)

        _broadcast(position_ids)
        batch = {
            'tokens': tokens,
            'labels': labels,
            'loss_mask': loss_mask,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'num_seqs': num_seqs
        }

    return batch