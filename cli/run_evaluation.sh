#!/bin/bash
#SBATCH --job-name=evaluation # create a short name for your job
#SBATCH --nodes=1 # node count
#SBATCH --ntasks-per-node=1 # total number of tasks across all nodes
#SBATCH --cpus-per-task=16 # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=8G # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:hgx:1 # number of gpus per node
#SBATCH -p pos # number of gpus per node
#SBATCH -o tts_evaluation_%j_%x.log

step=14000
model_path=/cognitive_comp/ccnl_common_data/wangrui/alm_sft_training/20250410/train/checkpoint/xpo-mcore-qwen2.5-7B-lr-8e-7-minlr-5e-7-bs-6-gbs-120-seqlen-4096-pr-bf16-tp-2-pp-4-cp-1-ac-false-do-true-sp-true-ti-18000-wi-66/iter_00014000_hf
model_config=/cognitive_comp/common_checkpoint/S_model_management/codec/20241017/Qwen2.5-7B-Codec0927-S204000-AEdivided50/codec_config.json 
ckpt_config=/cognitive_comp/common_checkpoint/S_model_management/codec/20241017/Qwen2.5-7B-Codec0927-S204000-AEdivided50/

text='天啊！这竟然是真的？我简直不敢相信！'
ref_text='求求你，再给我一次机会，我保证不会让你失望……'
ref_audio_path='./ref.mp3'

temperature=0.9
top_p=0.9
seed=0



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
