''' Train LLM on source code data.
    author: Daniel Nichols
    date: October 2022
'''
# std imports
from argparse import ArgumentParser
from typing import Iterable
from os import PathLike

# tpl imports
from datasets import load_dataset

# local imports
from load_dataset import get_source_filenames, get_source_file_size, get_loc, filter_bad_encoding, filter_duplicates


def get_args():
    ''' Parse the command line arguments and return the object with them as properties.
    '''
    parser = ArgumentParser(description='Train a LLM on source code data')
    parser.add_argument('--root', type=str, required=True, help='root of textual source data')
    parser.add_argument('--deduplicate', action='store_true', help='If provided, then data will be deduplicated')
    parser.add_argument('--model', type=str, choices=['NeoX', 'GPT2'], help='What model to train')
    return parser.parse_args()


def print_source_file_stats(fnames: Iterable[PathLike]):
    ''' Print meta-data about source files such as # files, LOC, and memory size.
    
        Args:
            fnames: File names to compute statistics over
    '''
    loc = get_loc(fnames)
    size = get_source_file_size(fnames)

    print('# source files: {:,}'.format(len(fnames)))
    print('LOC: {:,}'.format(loc))
    print('Dataset size: {:.3g} GB'.format(size / (1<<30)))


def main():
    args = get_args()

    fnames = get_source_filenames(args.root)
    fnames = filter_duplicates( filter_bad_encoding(fnames) )
    print_source_file_stats(fnames)

    dataset = load_dataset("text", name='HPC Source Dataset', data_files=fnames, encoding='utf-8')
    print(dataset)



if __name__ == '__main__':
    main()

