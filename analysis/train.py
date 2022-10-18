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
from torch.utils.data import DataLoader
from torch.optim import AdamW
from datasets import load_dataset, DatasetDict, load_from_disk
from tokenizers import Tokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM
from accelerate import Accelerator
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
    parser.add_argument('--load-tokens', type=str, help='retrieve tokens rather than retokenize')
    parser.add_argument('--model', type=str, default='gpt2', help='what model to train')
    parser.add_argument('--gradient-checkpointing', action='store_true', help='checkpoint gradients')
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
    if training_task == 'causal':
        model = AutoModelForCausalLM.from_pretrained(model_name)
    elif training_task == 'masked':
        model = AutoModelForMaskedLM.from_pretrained(model_name)

    return model


def train(dataset, model, batch_size=8):
    ''' Train model on dataset.

        Args:
            dataset: HuggingFace text dataset
            model: LLM
    '''
    accelerator = Accelerator()

    dataset.set_format('torch')
    train_dl = DataLoader(dataset['train'], shuffle=True, batch_size=batch_size)
    optimizer = AdamW(model.parameters(), lr=1e-5)

    train_dl, model, optimizer = accelerator.prepare(train_dl, model, optimizer)
    
    model.train()
    completed_steps = 0
    for step, batch in enumerate(dataset, start=1):
        loss = model(batch, labels=batch, use_cache=False).loss
        loss = loss / 1.0
        accelerator.backward()


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
    dataset = get_dataset(args.input)
    print(dataset)
    
    # tokenizer dataset
    logging.info('Tokenizing dataset...')
    if args.load_tokens:
        tokenized_dataset = load_from_disk(args.load_tokens)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        def tokenize_func(x):
            return tokenizer(x['text'], truncation=True, max_length=args.max_seq_length)
        
        tokenized_dataset = dataset.map(tokenize_func, batched=True)
        if args.save_tokens:
            tokenized_dataset.save_to_disk(args.save_tokens)

    # initialize model
    logging.info('Creating model...')
    model = get_model(args.model, training_task = args.lm_task)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # train
    logging.info('Training...')
    train(tokenized_dataset, model, accelerator)



if __name__ == '__main__':
    main()

