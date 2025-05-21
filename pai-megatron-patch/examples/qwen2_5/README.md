# Pretraining and SFT

## Installation

 Obtain the official Alibaba Cloud image. For detailed installation instructions, refer to [link](http://192.168.40.122/cognitive-computing/pai-megatron-patch/-/tree/dev_github/examples/qwen2_5).`dsw-registry.cn-wulanchabu.cr.aliyuncs.com/pai/pai-megatron-patch:24.07` 



## Model Download
Our foundation model is built upon Qwen-2.5 with an expanded audio vocabulary.

```bash
cd /mnt
mkdir qwen-ckpts
cd qwen-ckpts

git clone https://huggingface.co/IDEA-Emdoor/Qwen2.5-7B-ExtVocab

```

## Model Training Workflow
### Model format conversion
run `hf2mcore_qwen2.5_convertor.sh` shell.

```bash
cd /workspace/Pai-Megatron-Patch/toolkits/model_checkpoints_convertor/qwen

bash hf2mcore_qwen2.5_convertor.sh
```

```
MODEL_SIZE=$1                  # Model parameter: 0.5B/1.5B/3B/7B/14B/32B/72B
SOURCE_CKPT_PATH=$2            # Source path
TARGET_CKPT_PATH=$3            # Target path
TP=$4                          # Model parallelism degree
PP=$5                          # Pipeline parallelism degree
PR=$6                          # Precision conversion
USE_TE=$7                      # Whether to use Transformer Engine for modeling
mg2hf=$8                       # Whether to perform mcore2hf conversion
HG_CKPT_PATH=$9                # Path for HF checkpoint

```

### Pretraining & SFT
#### Pretraining and SFT Instruction Description
```
ENV=$1                          # Environment configuration switch: "dsw" for single-node training, "dlc" for multi-node training  
MODEL_SIZE=$2                   # Model parameter scale: 0.5B/1.5B/3B/7B/14B/32B/72B  
BATCH_SIZE=$3                   # Number of samples per iteration within a single data parallel group  
GLOBAL_BATCH_SIZE=$4            # Total number of samples per iteration across all data parallel groups  
LR=$5                           # Learning rate  
MIN_LR=$6                       # Minimum learning rate  
SEQ_LEN=$7                      # Sequence length  
PAD_LEN=$8                      # Padding length  
PR=$9                           # Training precision: fp16, bf16, fp8  
TP=${10}                        # Tensor parallelism degree  
PP=${11}                        # Pipeline parallelism degree  
CP=${12}                        # Context parallelism degree  
SP=${13}                        # Whether to use sequence parallelism: true, false  
DO=${14}                        # Whether to use Megatron-style ZeRO-1 optimizer for memory reduction: true, false  
FL=${15}                        # Whether to prioritize Flash Attention: true, false  
SFT=${16}                       # Whether to perform fine-tuning training: true, false  
AC=${17}                        # Activation checkpointing mode: sel (selective), full, offload, false  
OPTIMIZER_OFFLOAD=${18}         # Whether to enable optimizer offloading: false, static, auto  
SAVE_INTERVAL=${19}             # Checkpoint saving interval  
DATASET_PATH=${20}              # Training dataset path  
VALID_DATASET_PATH=${21}        # Validation dataset path  
PRETRAIN_CHECKPOINT_PATH=${22}  # Pre-trained model checkpoint path  
TRAIN_TOKENS_OR_ITERS=${23}     # Training tokens or iterations  
WARMUP_TOKENS_OR_ITERS=${24}    # Warmup tokens or iterations  
OUTPUT_BASEPATH=${25}           # Path for training output logs  

```
#### Pretraining

data parepare
```
cd pai-megatron-patch/toolkits/pretrain_data_preprocessing

1. The data sample should be prepared in the following format, refer to demo.parquet.

2. bash scripts/run_pretraining.sh
```

run `run_mcore_qwen.sh` shell.

```bash
cd pai-megatron-patch/examples/qwen2_5

sbatch run_pretraining.sh
```

#### SFT
data parepare
```
cd pai-megatron-patch/toolkits/sft_data_preprocessing

1. The data sample should be prepared in the following format, refer to demo.jsonl.

2. bash scripts/run_sft.sh
```
run `run_sft.sh` shell.

```bash
sbatch run_sft.sh

```