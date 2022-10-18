#!/bin/bash
#SBATCH -N 1
#SBATCH -t 01:00:00
#SBATCH -J hpc-llm-train

# config params
#DATASET="scratch-fnames.pkl"
DATASET="/scratch/zt1/project/bhatele-lab/user/dnicho/code-ml/data/dataset.jsonl"
TOKENIZER="gpt2"
MODEL="gpt2"
SEQ_LENGTH="1024"
LM_TASK="causal"

# setup environment
module load python/3.8.12/zen2
source .env/bin/activate

# run job
python train.py --input $DATASET --log info \
    --tokenizer $TOKENIZER \
    --model $MODEL \
    --lm-task $LM_TASK \
    --max-seq-length $SEQ_LENGTH 
