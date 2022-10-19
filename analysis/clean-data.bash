#!/bin/bash
# Data cannot persist in scratch space after training job. Clear these out from login node.
# author: Daniel Nichols
# date: October 2022

if [ $# -eq 1 ]; then
    DST="$1"
else
    DST="/scratch/zt1/project/bhatele-lab/user/dnicho/code-ml/data/dataset.jsonl"
fi

# remove dataset
echo "rm $DST"
rm $DST

# remove cache
echo "rm -r /scratch/zt1/project/bhatele-lab/user/dnicho/.cache/huggingface/datasets/*"
rm -r /scratch/zt1/project/bhatele-lab/user/dnicho/.cache/huggingface/datasets/*