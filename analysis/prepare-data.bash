#!/bin/bash
# Initial setup for training requires some work to be done on login node. Namely the dataset has to be copied
# to scratch space for the job to access it.
# author: Daniel Nichols
# date: October 2022

SRC="/afs/shell.umd.edu/project/bhatele-lab/shared/hpc-llms/data/dataset.jsonl"
DST="/scratch/zt1/project/bhatele-lab/user/dnicho/code-ml/data/dataset.jsonl"

echo "cp ${SRC} ${DST}"
cp ${SRC} ${DST}
