# BASE_PATH=${1}
BASE_PATH=/root/kl/distillm
MODEL_PATH=/root/autodl-tmp/model/hub

export TF_CPP_MIN_LOG_LEVEL=3

# only prompt for MiniLLM train
PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_dolly.py \
    --data-dir ${BASE_PATH}/data/self-inst/ \
    --processed-data-dir ${BASE_PATH}/processed_data/self-inst/prompt \
    --model-path /root/autodl-tmp/model/hub/models--openai-community--gpt2-large/snapshots/32b71b12589c2f8d625668d2335a01cac3249519 \
    --data-process-workers 32 \
    --max-prompt-length 256 \
    --dev-num 1000 \
    --only-prompt \
    --model-type gpt2

# prompt and response for baselines
PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_dolly.py \
    --data-dir ${BASE_PATH}/data/self-inst/ \
    --processed-data-dir ${BASE_PATH}/processed_data/self-inst/full \
    --model-path /root/autodl-tmp/model/hub/models--openai-community--gpt2-large/snapshots/32b71b12589c2f8d625668d2335a01cac3249519 \
    --data-process-workers 32 \
    --max-prompt-length 256 \
    --dev-num 1000 \
    --model-type gpt2
