#!/bin/bash
#SBATCH -N 1
#SBATCH -t 06:00:00
#SBATCH -J train-causal
#SBATCH -A bhatele-lab-cmsc
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=196608

DATASET="daniellnichols/hpc-source"
MODEL="daniellnichols/gpt-neo-hpc-source"
CACHE_DIR="/scratch/zt1/project/bhatele-lab/user/dnicho/.cache/huggingface"
OUTPUT_DIR="/scratch/zt1/project/bhatele-lab/user/dnicho/code-ml/data/gpt-neo-hpc-ckpt"
DSCONFIG="./ds_config_zero2.json"
HUB_TOKEN="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"

MAX_STEPS="25000"

module load python/3.8.12/zen2 git-lfs/zen2/3.1.2 openmpi/4.1.1/gcc/9.4.0/zen2 cuda/11.6.2/gcc
source .env/bin/activate

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${CUDA_HOME}/lib64"

echo "device(s): $CUDA_VISIBLE_DEVICES"

deepspeed run_clm.py \
    --model_name_or_path EleutherAI/gpt-neo-2.7B \
    --tokenizer_name ./hpc-tok \
    --output_dir $OUTPUT_DIR \
    --dataset_name $DATASET \
    --validation_split_percentage 5 \
    --cache_dir $CACHE_DIR \
    --optim adamw_torch \
    --fp16 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --preprocessing_num_workers 1 \
    --seed 42 \
    --do_eval \
    --do_train \
    --deepspeed $DSCONFIG \
    --max_steps $MAX_STEPS \
    --save_steps 1000 \
    --evaluation_strategy steps \
    --eval_steps 1000 \
    --max_eval_samples 1000 \
    --logging_steps 50 \
    --log_level passive \
    --use_auth_token \
    --push_to_hub \
    --hub_model_id $MODEL \
    --hub_private_repo \
    --hub_token $HUB_TOKEN
