# HPC LLM Training

This repo contains scripts for training LLMs on HPC source code data.
It is organized as follows:

- ***data:*** scripts and utilities for collecting and preprocessing the dataset [[README]](data/README.md)
- ***analysis:*** scripts related to analyzing dataset and training LLM [[README]](analysis/README.md)

To overview of the workflow from start to finish is as follows.
Use `data/collect-repo-metadata.py` and `data/edit-metadata.py` to create dataset of GitHub repositories, if desired,
otherwise use the existing `repos-gt3.csv` dataset.
Run `data/clone-repos.py` to clone the repositories to a desired location.
`data/collect-dataset.py` script can then be used to create a json lines dataset with all the textual data.
The `analysis/run_clm-*.sbatch` scripts can then be used to train the models on the data.

## Notes and Misc.

*Weird bug fix:*
I had to change the default value of `max_workers` from 64 to 32 in the parameter list of 
`_get_origin_metadata_locally_or_by_urls` in `datasets/data_files.py#L708`. 
Zaratan's CPUs have 64 cores, but tqdm errors trying to start 64 threads for some reason.
Similar to this I also generally have to `export TOKENIZERS_PARALLELISM=false` to prevent 
huggingface from spinning up threads in forked processes.