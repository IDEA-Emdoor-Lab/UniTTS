#!/bin/bash
#SBATCH --job-name=mmap # create a short name for your job
#SBATCH --nodes=1 # node count
#SBATCH --ntasks-per-node=1 # total number of tasks across all nodes
#SBATCH --cpus-per-task=30 # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=32G # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:hgx:1 # number of gpus per node
#SBATCH -p pos # number of gpus per node

#SBATCH -o ./log/make_mmap_gpu.log


CURRENT_DIR="$( cd "$( dirname "$0" )" && pwd )"
MEGATRON_PATH=$( dirname $( dirname ${CURRENT_DIR}))
export PYTHONPATH=$PYTHONPATH:${MEGATRON_PATH}:${MEGATRON_PATH}/PAI-Megatron-LM-240718

model_dir=$1 # model path

input_data_path=./data/sft_train.jsonl
tokenizer=Qwen2Tokenizer
seq_len=8192
output_data_path=./output/sft

default_packing=true

if [ -z ${default_packing} ]; then
  default_packing=false
fi

if [ $default_packing = true ]; then
  packing_option="\
    --sequence-packing 
  "
else
  packing_option=""
fi

cmd="python build_idxmap_sft_dataset.py \
  --input ${input_data_path} \
  --output-prefix ${output_data_path} \
  --patch-tokenizer-type ${tokenizer} \
  --load ${model_dir} \
  --seq-length ${seq_len} \
  --workers 1 \
  --partitions 1 ${packing_option} \
  --split_size 9000 "

echo $cmd
eval $cmd

ELAPSED_TIME=$(($SECONDS - $START_TIME))
echo "$(($ELAPSED_TIME/60)) min $(($ELAPSED_TIME%60)) sec"
