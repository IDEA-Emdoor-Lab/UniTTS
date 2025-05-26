#!/bin/bash


model_path=$1
model_config=$2 
ckpt_config=$3

use_vllm=true

export GRADIO_TEMP_DIR=/cognitive_comp/sunqianguo/workspace/pai-megatron-patch/toolkits/webdemo
CUDA_VISIBLE_DEVICES=6 python webdemo_audio_english.py \
    --host 0.0.0.0 \
    --port 8893 \
    --model_config $model_config \
    --ckpt_config $ckpt_config \
    --model_path $model_path \
    --demo_url /demo/541833 \
    --api_url /api/demo/541832 \
    --use_vllm $use_vllm



