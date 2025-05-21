#!/bin/bash
#SBATCH --job-name=sft7B
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:hgx:8
#SBATCH --cpus-per-task=80
#SBATCH --mem-per-cpu=20G
#SBATCH -p pos

#SBATCH -o ./log/train7B_sft_%j_%x.log

set -e
export TORCH_DISTRIBUTED_DEBUG=DETAIL
ENV=dsw
CURRENT_DIR="$( cd "$( dirname "$0" )" && pwd )"
MEGATRON_PATH=$( dirname $( dirname ${CURRENT_DIR}))
MEGATRON_PATH=/cognitive_comp/sunqianguo/workspace/tmp/Pai-megatron-patch-sft/pai-megatron-patch/
PYTHONPATH=$(which python)
export PYTHONPATH=${MEGATRON_PATH}:${MEGATRON_PATH}/PAI-Megatron-LM-240718:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

wandb_project="alm-sft-20250208"
wandb_team="ccnl-s-team"
wandb_exp_name="alm-sft"

DOCKER_PATH=$1 # The input Docker image must be in .sif format.
DATASET_PATH1=$2 # train data path 
VALID_DATASET_PATH=$3 # valid data path
PRETRAIN_CHECKPOINT_PATH=$4 # load model path, megatron
OUTPUT_BASEPATH=$5 # save checkpoints path

mkdir -p ${OUTPUT_BASEPATH}

# Here are some configs controled by env
if [ -z ${MP_DATASET_TYPE} ];then
    MP_DATASET_TYPE="Idxmap"
fi

if [ -z ${MP_AC_LAYERS} ];then
    MP_AC_LAYERS=1
fi

if [ $ENV = dsw ]; then
    echo "dsw"
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export NCCL_SOCKET_IFNAME=bond0

    nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
    nodes_array=($nodes)
    head_node=${nodes_array[0]}
    head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
    #head_node_ip=10.10.13.26
    MASTER_ADDR=${head_node_ip}
    MASTER_PORT=$(shuf -n 1 -i 40000-65535)
    echo ${head_node_ip}
    echo $MASTER_PORT
    NNODES=1
    NODE_RANK=0
    GPUS_PER_NODE=8

elif [ $ENV = dlc ]; then
    NNODES=${WORLD_SIZE}
    NODE_RANK=${RANK}
    GPUS_PER_NODE=${KUBERNETES_CONTAINER_RESOURCE_GPU}
fi

if [ -z ${MP_VP} ]; then
    vp_options=""
else
    vp_options=" \
        --num-layers-per-virtual-pipeline-stage ${MP_VP}"
fi

if [ -z ${MP_SFT_PACKING} ]; then
    MP_SFT_PACKING=false
fi


DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --rdzv_id $RANDOM --rdzv_backend c10d --rdzv_endpoint ${MASTER_ADDR}:${MASTER_PORT}"
echo $DISTRIBUTED_ARGS

### BASE CONFIG ###
MODEL_SIZE=7B
BATCH_SIZE=2
GLOBAL_BATCH_SIZE=32
LR=9e-6
MIN_LR=5e-6
SEQ_LEN=8192
PAD_LEN=8192
PR=bf16
### BASE CONFIG ###

### PARALLEL / BOOL OPTION ###
TP=2
PP=4
CP=1
SP=true
DO=true
FL=true
SFT=true
### PARALLEL / BOOL OPTION ###


### OTHERS ###
AC=false
OPTIMIZER_OFFLOAD=false

SAVE_INTERVAL=300

# the following two values will not be used when SFT is true
TRAIN_TOKENS=130000
WARMUP_TOKENS=1000  #1000000000 
############################### 



if [ $FL = true ]; then
    export NVTE_FLASH_ATTN=1 NVTE_FUSED_ATTN=0
elif [ $FL = false ]; then
    export NVTE_FLASH_ATTN=0 NVTE_FUSED_ATTN=1
fi

if [ $MODEL_SIZE = 0.5B ]; then

NUM_LAYERS=24
HIDDEN_SIZE=896
NUM_ATTN_HEADS=14
INTERMEDIATE_SIZE=4864
NUM_KEY_VALUE_HEADS=2
MAX_POSITION_EMBEDDINGS=32768
EXTRA_VOCAB_SIZE=293
RMS_NORM_EPS=1e-6
gqa_options=" \
		    --group-query-attention \
		    --num-query-groups ${NUM_KEY_VALUE_HEADS}"


tie_option=""

elif [ $MODEL_SIZE = 1.5B ]; then

NUM_LAYERS=28
HIDDEN_SIZE=1536
NUM_ATTN_HEADS=12
INTERMEDIATE_SIZE=8960
NUM_KEY_VALUE_HEADS=2
MAX_POSITION_EMBEDDINGS=32768
EXTRA_VOCAB_SIZE=293
RMS_NORM_EPS=1e-6
gqa_options=" \
		    --group-query-attention \
		    --num-query-groups ${NUM_KEY_VALUE_HEADS}"

tie_option=""

elif [ $MODEL_SIZE = 3B ]; then

NUM_LAYERS=36
HIDDEN_SIZE=2048
NUM_ATTN_HEADS=16
INTERMEDIATE_SIZE=11008
NUM_KEY_VALUE_HEADS=2
MAX_POSITION_EMBEDDINGS=32768
EXTRA_VOCAB_SIZE=293
RMS_NORM_EPS=1e-6
gqa_options=" \
		    --group-query-attention \
		    --num-query-groups ${NUM_KEY_VALUE_HEADS}"

tie_option=""

elif [ $MODEL_SIZE = 7B ]; then

NUM_LAYERS=28
HIDDEN_SIZE=3584
NUM_ATTN_HEADS=28
INTERMEDIATE_SIZE=18944
NUM_KEY_VALUE_HEADS=4
MAX_POSITION_EMBEDDINGS=131072
EXTRA_VOCAB_SIZE=33197  # 421 + 32768 = 33189
RMS_NORM_EPS=1e-6
gqa_options=" \
		    --group-query-attention \
		    --num-query-groups ${NUM_KEY_VALUE_HEADS}"

tie_option=" \
        --untie-embeddings-and-output-weights \
        "

elif [ $MODEL_SIZE = 14B ]; then

NUM_LAYERS=48
HIDDEN_SIZE=5120
NUM_ATTN_HEADS=40
INTERMEDIATE_SIZE=13824
NUM_KEY_VALUE_HEADS=8
MAX_POSITION_EMBEDDINGS=131072
EXTRA_VOCAB_SIZE=421
RMS_NORM_EPS=1e-5
gqa_options=" \
		    --group-query-attention \
		    --num-query-groups ${NUM_KEY_VALUE_HEADS}"

tie_option=" \
        --untie-embeddings-and-output-weights \
        "
elif [ $MODEL_SIZE = 32B ]; then

NUM_LAYERS=64
HIDDEN_SIZE=5120
NUM_ATTN_HEADS=40
INTERMEDIATE_SIZE=27648
NUM_KEY_VALUE_HEADS=8
MAX_POSITION_EMBEDDINGS=131072
EXTRA_VOCAB_SIZE=421
RMS_NORM_EPS=1e-5
gqa_options=" \
		    --group-query-attention \
		    --num-query-groups ${NUM_KEY_VALUE_HEADS}"

tie_option=" \
        --untie-embeddings-and-output-weights \
        "
elif [ $MODEL_SIZE = 72B ]; then

NUM_LAYERS=80
HIDDEN_SIZE=8192
NUM_ATTN_HEADS=64
INTERMEDIATE_SIZE=29568
NUM_KEY_VALUE_HEADS=8
MAX_POSITION_EMBEDDINGS=131072
EXTRA_VOCAB_SIZE=421
RMS_NORM_EPS=1e-5
gqa_options=" \
		    --group-query-attention \
		    --num-query-groups ${NUM_KEY_VALUE_HEADS}"

tie_option=" \
        --untie-embeddings-and-output-weights \
        "

fi

TP_COMM_OVERLAP=$(( ($TP > 1) ? 1 : 0 ))
comm_overlap_option="\
    --overlap-grad-reduce \
    --overlap-param-gather"
 

if [ $TP_COMM_OVERLAP -eq 1 ]; then
    comm_overlap_option="\
        --tp-comm-overlap \
        --overlap-grad-reduce \
        --overlap-param-gather"
fi

if [ $AC = full ]; then
    _check=$(( ($NUM_LAYERS / $PP) % ${MP_AC_LAYERS} ))
    if [ $_check != 0 ]; then
        echo "the num layers per pp rank must be a multiple of the recompute layers."
        exit -1
    fi
    activation_checkpoint_options=" \
		    --recompute-method uniform \
            --recompute-num-layers ${MP_AC_LAYERS} \
		    --recompute-granularity full"
elif [ $AC = sel ]; then
    activation_checkpoint_options=" \
        --recompute-activations"
elif [ $AC = none ]; then
    activation_checkpoint_options=" \
    "
elif [ $AC = offload ]; then
    activation_checkpoint_options=" \
		    --cpu-offloading \
		    --cpu-offloading-num-layers ${MP_AC_LAYERS}"
    if [ $TP_COMM_OVERLAP -eq 1 ]; then
        echo "Disable --overlap-grad-reduce and --overlap-param-gather when cpu offloading is on..."
        comm_overlap_option="\
            --tp-comm-overlap"
    else
        echo "Disable --overlap-grad-reduce and --overlap-param-gather when cpu offloading is on..."
        comm_overlap_option=""
    fi
fi

if [ $PR = fp16 ]; then
    pr_options=" \
		    --fp16 \
            --apply-query-key-layer-scaling"
    export NVTE_APPLY_QK_LAYER_SCALING=1
elif [ $PR = bf16 ]; then
    pr_options=" \
        --bf16"
elif [ $PR = fp8 ]; then
    pr_options=" \
        --bf16 \
        --fp8-format hybrid \
        --fp8-amax-compute-algo max \
        --fp8-amax-history-len 1024"
fi

if [ $OPTIMIZER_OFFLOAD != false ] && [ $DO = false ]; then
    echo "Offload optimizer is valid only if \$DO=true"
    DO=true
fi

if [ $DO = true ]; then
    do_options=" \
		    --use-distributed-optimizer"

elif [ $DO = false ]; then
    do_options=" \
                    "
fi

te_options=" \
        --transformer-impl transformer_engine"

if [ $SP = true ] && [ $TP -gt 1 ]; then
    sp_options=" \
		    --sequence-parallel"

elif [ $SP = false ]; then
    sp_options=" \
                    "
fi

if [ $PRETRAIN_CHECKPOINT_PATH != none ]; then
    load_options=" \
            --load $PRETRAIN_CHECKPOINT_PATH"
fi

if [ $OPTIMIZER_OFFLOAD = 'static' ]; then
    offload_option=" \
        --optimizer hybridadam \
        --optimizer-offload-policy static \
        --optimizer-offload-fraction 1.0"
elif [ $OPTIMIZER_OFFLOAD = 'auto' ]; then
    offload_option=" \
        --optimizer hybridadam \
        --optimizer-offload-policy auto"
else
    offload_option=""
fi

if [ $SFT = true ]; then
    TRAIN_ITERS=3015
    LR_WARMUP_ITERS=10
    LR_DECAY_ITERS=$(( ${TRAIN_ITERS} - ${LR_WARMUP_ITERS}))
    PREFIX="finetune-mcore-qwen2.5-${MODEL_SIZE}-lr-${LR}-minlr-${MIN_LR}-bs-${BATCH_SIZE}-gbs-${GLOBAL_BATCH_SIZE}-seqlen-${SEQ_LEN}"
    sft_option=" \
         --eod-mask-loss \
         --train-mode finetune"
else
    TRAIN_ITERS=$(( ${TRAIN_TOKENS} / ${GLOBAL_BATCH_SIZE} / ${SEQ_LEN} ))
    LR_WARMUP_ITERS=$(( ${WARMUP_TOKENS}  / ${GLOBAL_BATCH_SIZE} / ${SEQ_LEN} ))
    LR_DECAY_ITERS=$(( ${TRAIN_TOKENS} /  ${GLOBAL_BATCH_SIZE} / ${SEQ_LEN} ))
    PREFIX="pretrain-mcore-qwen2.5-${MODEL_SIZE}-lr-${LR}-minlr-${MIN_LR}-bs-${BATCH_SIZE}-gbs-${GLOBAL_BATCH_SIZE}-seqlen-${SEQ_LEN}"
    sft_option=" \
        --train-mode pretrain"
fi

if [ ${MP_DATASET_TYPE} = "raw" ]; then
    dataset_option=" \
        --train-data-path ${DATASET_PATH1} \
        --valid-data-path ${VALID_DATASET_PATH} \
        --dataloader-type cyclic \
        --dataset Qwen-SFT-Raw"
else 
    dataset_option=" \
        --data-path ${DATASET_PATH1} \
        --split 99,1,0 \
        --dataset LLama-Pretrain-Idxmap"
fi

if [ ${MP_SFT_PACKING} = true ]; then
    packing_options=" \
        --reset-position-ids \
        --no-create-attention-mask-in-dataloader
    "
else
    packing_options=""
fi

##### Prepare logdirs #######
NAME="${PREFIX}-pr-${PR}-tp-${TP}-pp-${PP}-cp-${CP}-ac-${AC}-do-${DO}-sp-${SP}-ti-${TRAIN_ITERS}-wi-${LR_WARMUP_ITERS}"
mkdir -p "${OUTPUT_BASEPATH}/tensorboard/"
mkdir -p "${OUTPUT_BASEPATH}/checkpoint/"
mkdir -p "${OUTPUT_BASEPATH}/log/"
current_time=$(date "+%Y.%m.%d-%H.%M.%S")
#TENSORBOARD_DIR="${OUTPUT_BASEPATH}/tensorboard/${NAME}_${current_time}"
TENSORBOARD_DIR="${OUTPUT_BASEPATH}/tensorboard/"
mkdir -p ${TENSORBOARD_DIR}
SAVED_PRETRAIN_CHECKPOINT_PATH="${OUTPUT_BASEPATH}/checkpoint/${NAME}"

mkdir -p ${SAVED_PRETRAIN_CHECKPOINT_PATH}
find -L ${PRETRAIN_CHECKPOINT_PATH} -maxdepth 1 -type f -name "*.json" -print0 | xargs -0 cp -t ${SAVED_PRETRAIN_CHECKPOINT_PATH}
find -L ${PRETRAIN_CHECKPOINT_PATH} -maxdepth 1 -type f -name "merges.txt" -print0 | xargs -0 cp -t ${SAVED_PRETRAIN_CHECKPOINT_PATH}

megatron_options="  \
        --save ${SAVED_PRETRAIN_CHECKPOINT_PATH} \
        --lr ${LR} \
        --min-lr ${MIN_LR} \
        --lr-decay-style cosine \
        --weight-decay 0.1 \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --clip-grad 1.0 \
        --init-method-std 0.008 \
        --attention-dropout 0.0 \
        --hidden-dropout 0.0 \
        --lr-decay-iters ${LR_DECAY_ITERS} \
        --lr-warmup-iters ${LR_WARMUP_ITERS} \
        --train-iters ${TRAIN_ITERS} \
        --micro-batch-size ${BATCH_SIZE} \
        --global-batch-size ${GLOBAL_BATCH_SIZE} \
        --num-layers ${NUM_LAYERS} \
        --hidden-size ${HIDDEN_SIZE} \
        --num-attention-heads ${NUM_ATTN_HEADS} \
        --ffn-hidden-size ${INTERMEDIATE_SIZE} \
        --seq-length ${SEQ_LEN} \
        --max-position-embeddings ${MAX_POSITION_EMBEDDINGS} \
        --max-padding-length ${PAD_LEN} \
        --log-interval 1 \
        --log-throughput \
        --eval-interval 10000 \
        --eval-iters 10 \
        --save-interval ${SAVE_INTERVAL} \
        --tensorboard-queue-size 1 \
        --tensorboard-dir ${TENSORBOARD_DIR} \
        --log-timers-to-tensorboard \
        --log-batch-size-to-tensorboard \
        --log-validation-ppl-to-tensorboard \
        --tensor-model-parallel-size ${TP} \
        --pipeline-model-parallel-size ${PP} \
        --context-parallel-size ${CP} \
        --num-workers 8 \
        --extra-vocab-size ${EXTRA_VOCAB_SIZE} \
        --patch-tokenizer-type Qwen2Tokenizer \
        --swiglu \
        --normalization RMSNorm \
        --norm-epsilon ${RMS_NORM_EPS} \
        --use-rotary-position-embeddings \
        --position-embedding-type rope \
        --disable-bias-linear \
        --add-qkv-bias \
        --rotary-percent 1.0 \
        --rotary-base 1000000 \
        --rotary-seq-len-interpolation-factor 1 \
        --reset-attention-mask \
        --distributed-timeout-minutes 3600 \
        --no-load-optim \
        --no-load-rng \
        --wandb-project ${wandb_project} \
        --wandb-team ${wandb_team} \
        --wandb-exp-name ${wandb_exp_name} \
        "

#--wandb-project ${wandb_project} \
#        --wandb-team ${wandb_team} \
#        --wandb-exp-name ${wandb_exp_name} \

# --no-save-optim \
#--no-load-optim \
#--no-load-rng \

# cd /cognitive_comp/wangrui/codes/tmp/Pai-Megatron-Patch/examples/qwen2_5/
run_cmd="torchrun $DISTRIBUTED_ARGS ../qwen2/pretrain_qwen.py
 ${megatron_options} ${dataset_option} ${pr_options} ${load_options} ${te_options} ${activation_checkpoint_options} \
 ${do_options} ${sp_options} ${gqa_options} ${offload_option} ${comm_overlap_option} ${sft_option}  ${tie_option} ${vp_options} ${packing_options}"


srun singularity exec \
    --env RUST_BACKTRACE=full \
    --env PYTHONPATH=${MEGATRON_PATH}:${MEGATRON_PATH}/PAI-Megatron-LM-240718:$PYTHONPATH \
    --nv \
    -B /cognitive_comp/:/cognitive_comp/ \
    -B /home/:/home/ \
    $DOCKER_PATH \
    ${run_cmd}
#${run_cmd}
echo ${run_cmd}
#eval ${run_cmd}
set +x
