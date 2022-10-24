#!/bin/bash
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH -J train-causal
#SBATCH -A bhatele-lab-cmsc
#SBATCH -p gpu
#SBATCH --gres=gpu:a100

#DATASET="/scratch/zt1/project/bhatele-lab/user/dnicho/code-ml/data/dataset.json"
DATASET="daniellnichols/hpc-source"
MODEL="daniellnichols/gpt2-hpc-source"
CACHE_DIR="/scratch/zt1/project/bhatele-lab/user/dnicho/.cache/huggingface"
OUTPUT_DIR="/scratch/zt1/project/bhatele-lab/user/dnicho/code-ml/data/gpt2-hpc-ckpt"

module load python/3.8.12/zen2
source .env/bin/activate

python run_clm.py \
    --model_name_or_path gpt2 \
    --tokenizer_name ./hpc-tok \
    --output_dir $OUTPUT_DIR \
    --dataset_name $DATASET \
    --validation_split_percentage 5 \
    --cache_dir $CACHE_DIR \
    --optim adamw_hf \
    --num_train_epochs 1 \
    --fp16 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 1 \
    --seed 42 \
    --do_eval \
    --do_train \
    --max_eval_samples 1000 \
    --eval_steps 1000 \
    --dataloader_num_workers 1 \
    --logging_dir ${OUTPUT_DIR}/logging \
    --logging_steps 250 \
    --log_level passive

    #--use_auth_token \
    #--push_to_hub \
    #--hub_model_id $MODEL \
    #--hub_private_repo 
