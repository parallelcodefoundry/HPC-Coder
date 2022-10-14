# analysis

This directory contains scripts for analyzing/plotting statistics and metadata 
about the dataset. It also contains the scripts for training LLMs on the 
dataset.

## repo-plots.py and generate-all-repo-plots.bash
`repo-plots.py` is for generating figures with summary stats about the dataset.
Run `python repo-plots.py -h` to see all of the options.
`generate-all-repo-plots.bash` is a helper script to generate all of the 
possible figures into the `figs/` directory.

## load_dataset.py
Not meant to be run on its own. This script contains a set of functions for 
interacting with the dataset, collecting source files, and processing them.

## train.py
This is the main script for training LLMs on the dataset. It uses the 
huggingface _transformers_ and _datasets_ libraries.

