# HPC LLM Training

This repo contains scripts for training LLMs on HPC source code data.
It is organized as follows:

- ***data:*** scripts and utilities for collecting and preprocessing the dataset [[README]](data/README.md)
- ***analysis:*** scripts related to analyzing dataset and training LLM [[README]](analysis/README.md)


## Notes and Misc.

*Weird bug fix:*
I had to change the default value of `max_workers` from 64 to 32 in the parameter list of 
`_get_origin_metadata_locally_or_by_urls` in `datasets/data_files.py#L708`. 
Zaratan's CPUs have 64 cores, but tqdm errors trying to start 64 threads for some reason.