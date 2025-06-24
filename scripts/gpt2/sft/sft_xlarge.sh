#! /bin/bash

# MASTER_ADDR=localhost
# MASTER_PORT=${2-2012}
# NNODES=1
# NODE_RANK=0
# GPUS_PER_NODE=${3-16}

# GPUS=(0,1)
GPUS=(0 1 2 3)
export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${GPUS[*]}")

MASTER_ADDR=localhost
MASTER_PORT=66$(($RANDOM%90+10))
NNODES=1
NODE_RANK=0
# GPUS_PER_NODE=${#GPUS[@]}
GPUS_PER_NODE=4

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

# model
BASE_PATH="/root/kl/distillm"
CKPT_NAME="gpt2-xlarge"
CKPT="/root/autodl-tmp/save/results/gpt2/train/sft/gpt2-xlarge/e10-bs2-lr5e-05-G1-N4-NN1/6250"
# CKPT="gpt2-xl" # download automatically
# data
DATA_DIR="${BASE_PATH}/processed_data/dolly/full/gpt2/"
# hp
BATCH_SIZE=2

EPOCHS=10
TRAIN_NUM=10000
SAVE_INTERVAL=8
GRAD_ACC=1
SAVE_INTERVAL_VAL=$(( TRAIN_NUM / BATCH_SIZE * EPOCHS / GRAD_ACC / SAVE_INTERVAL ))
LR=0.00005
# EVAL_BATCH_SIZE=8
EVAL_BATCH_SIZE=8
# length
MAX_LENGTH=512
# runtime
SAVE_PATH="/root/autodl-tmp/save/results/gpt2/train/sft/${CKPT_NAME}"
# seed
SEED=10
SEED_ORDER=10


OPTS=""
# model
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-path ${CKPT}"
OPTS+=" --ckpt-name ${CKPT_NAME}"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
OPTS+=" --gradient-checkpointing"
# data
OPTS+=" --data-dir ${DATA_DIR}"
OPTS+=" --num-workers 0"
OPTS+=" --dev-num 1000"
OPTS+=" --train-num ${TRAIN_NUM}"
# hp
OPTS+=" --lr ${LR}"
OPTS+=" --batch-size ${BATCH_SIZE}"
OPTS+=" --eval-batch-size ${EVAL_BATCH_SIZE}"
OPTS+=" --gradient-accumulation-steps ${GRAD_ACC}"
OPTS+=" --warmup-iters 0"
OPTS+=" --lr-decay-style cosine"
OPTS+=" --weight-decay 1e-2"
OPTS+=" --clip-grad 1.0"
OPTS+=" --epochs ${EPOCHS}"
# length
OPTS+=" --max-length ${MAX_LENGTH}"
OPTS+=" --max-prompt-length 256"
# runtime
OPTS+=" --do-train"
OPTS+=" --do-valid"
OPTS+=" --eval-gen"
# OPTS+=" --save-interval ${SAVE_INTERVAL_VAL}"
OPTS+=" --save-interval ${SAVE_INTERVAL_VAL}"
OPTS+=" --eval-interval -1"
OPTS+=" --log-interval 4"
OPTS+=" --mid-log-num -1"
OPTS+=" --save ${SAVE_PATH}"
# seed
OPTS+=" --seed ${SEED}"
OPTS+=" --seed-order ${SEED_ORDER}"
# deepspeed
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config.json"
# type
OPTS+=" --type lm"
# gen
OPTS+=" --do-sample"
OPTS+=" --top-k 0"
OPTS+=" --top-p 1.0"
OPTS+=" --temperature 1.0"


export NCCL_DEBUG=""
export WANDB_DISABLED=True
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}
CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/finetune.py ${OPTS} $@"

echo ${CMD}
echo "PYTHONPATH=${PYTHONPATH}"
mkdir -p ${SAVE_PATH}
${CMD}
