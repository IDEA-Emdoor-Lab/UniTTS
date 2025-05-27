#!/bin/bash
#SBATCH --job-name=evaluation # create a short name for your job
#SBATCH --nodes=1 # node count
#SBATCH --ntasks-per-node=1 # total number of tasks across all nodes
#SBATCH --cpus-per-task=16 # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=8G # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:hgx:1 # number of gpus per node
#SBATCH -p pos # number of gpus per node
#SBATCH -o text_conversation.log

model_path=$1 # UniTTS model path             example: ./IDEA-Emdoor/UniTTS-mixed-v0.1/
model_config=$2 # codec_model_config_path     example: ./DistilCodec-v1.0/model_config.json
ckpt_config=$3 # codec_ckpt_path              example: ./DistilCodec-v1.0

text='天啊！这竟然是真的？我简直不敢相信！'


temperature=0.75
top_p=0.75
seed=0
inference_type='text_conversation'

output_dir=output/
mkdir -p $output_dir

python inference.py \
    --model_config $model_config \
    --ckpt_config $ckpt_config \
    --model_name $model_path \
    --output_dir $output_dir \
    --temperature $temperature \
    --top_p $top_p \
    --seed $seed \
    --text $text \
    --ref_text $ref_text \
    --ref_audio_path $ref_audio_path \
    --inference_type $inference_type
