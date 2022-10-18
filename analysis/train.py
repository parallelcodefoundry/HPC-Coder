''' Train LLM on source code data.
    author: Daniel Nichols
    date: October 2022
'''
# std imports
from argparse import ArgumentParser
from typing import Iterable, Optional, Union
import logging
from os import PathLike, environ

# tpl imports
import torch
from datasets import load_dataset, DatasetDict
from tokenizers import Tokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM
import tqdm


def get_args():
    ''' Parse the command line arguments and return the object with them as properties.
    '''
    parser = ArgumentParser(description='Train a LLM on source code data')
    parser.add_argument('--log', choices=['INFO', 'DEBUG', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO', type=str.upper, help='logging level')
    parser.add_argument('--input', type=str, required=True, help='root of textual source data or path to pkl of ' +
        'filenames list')
    parser.add_argument('--save-tokens', type=str, help='path to store token data')
    parser.add_argument('--model', type=str, default='gpt2', help='what model to train')
    parser.add_argument('--lm-task', default='causal', choices=['causal', 'masked'], help='LM training objective')
    parser.add_argument('--tokenizer', type=str, default='gpt2', help='what text tokenizer to use')
    parser.add_argument('--max-seq-length', type=int, default=1024, help='maximum sequence length')
    return parser.parse_args()


def get_dataset(dataset_path: PathLike, name: str = 'HPC-Source-Dataset', type: str = 'json') -> DatasetDict:
    ''' Fetch the dataset from dataset_path and return a huggingface DatasetDict object. Currently this is just
        a light wrapper around `load_dataset`.

        Args:
            dataset_path: path to dataset
    '''
    return load_dataset(type, name=name, data_files=dataset_path)
    

def get_model(model_name: Union[str, PathLike], training_task: str = 'causal'):
    ''' Return the pretrained model from file or huggingface.

        Args:
            model_name: name of huggingface model or path to model
            training_task: causal or masked
    '''
    assert training_task in ['causal', 'masked']

    model = None
    if training_task == 'causual':
        model = AutoModelForCausalLM.from_pretrained(model_name)
    elif training_task == 'masked':
        model = AutoModelForMaskedLM.from_pretrained(model_name)

    return model


def train(dataset, model):
    ''' Train model on dataset.

        Args:
            dataset: HuggingFace text dataset
            model: LLM
    '''
    pass


def main():
    args = get_args()

    # setup logging
    numeric_level = getattr(logging, args.log.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: {}'.format(args.log))
    logging.basicConfig(format='%(asctime)s [%(levelname)s] -- %(message)s', 
        level=numeric_level) #filename='log.txt', filemode='w')

    # environment setup
    logging.info('Setting up environment...')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    environ['TOKENIZERS_PARALLELISM'] = '0'
    environ['OMP_NUM_THREADS'] = '32'
    #tqdm.tqdm.monitor_interval = 0  # fixes bug where tqdm calls in HF error due to monitor threading
    logging.info('Using device: {}'.format(device))

    # gather and initialize dataset
    logging.info('Creating dataset...')
    dataset = get_dataset(args.input, deduplicate=args.deduplicate, fnames_cache_output=args.cache_fnames,
        print_stats=args.dataset_info)
    print(dataset)
    
    # tokenizer dataset
    logging.info('Tokenizing dataset...')
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    def tokenize_func(x):
        return tokenizer(x['text'], truncation=True, max_length=args.max_seq_length)
    
    tokenized_dataset = dataset.map(tokenize_func, batched=True)
    if args.save_tokens:
        tokenized_dataset.save_to_disk(args.save_tokens)
    print(tokenized_dataset)

    # initialize model
    logging.info('Creating model...')
    model = get_model(args.model, training_task = args.lm_task)
    model.to(device)

    # train
    logging.info('Training...')
    train(tokenized_dataset, model)



if __name__ == '__main__':
    main()

