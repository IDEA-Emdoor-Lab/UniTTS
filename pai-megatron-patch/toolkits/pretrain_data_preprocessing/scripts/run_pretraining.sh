#!/bin/bash
#SBATCH --job-name=pretrain_mmap
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=30G
#SBATCH -p pos

#SBATCH -o ./log/make_pretrain_mmap.log

START_TIME=$SECONDS
postfix=".parquet"
file_tag=".jsonl"
decode_wav=false
MEGATRON_PATH=${MEGATRON_PATCH_PATH}/Megatron-LM-240405
export PYTHONPATH=${MEGATRON_PATH}:${MEGATRON_PATCH_PATH}:$PYTHONPATH

model_dir=$1 # model path

input_data_dir1=./data/
input_cache=./output/cache/tmp.json
output_data_dir=./output/


INPUT="${input_data_dir}"

mkdir -p ${output_data_dir}


python preprocess_data_multi_modal_v2.py \
--input_list ${input_data_dir1} \
--input_cache ${input_cache} \
--output-prefix ${output_data_dir}/pretrain_parquet_audio \
--dataset-impl mmap \
--patch-tokenizer-type Qwen2Tokenizer \
--load ${model_dir} \
--workers 10 \
--seq_length 8192 \
--append-eod \
--type parquet \
--postfix ${postfix} \
--file_tag ${file_tag} 




ELAPSED_TIME=$(($SECONDS - $START_TIME))
echo "$(($ELAPSED_TIME/60)) min $(($ELAPSED_TIME%60)) sec"
