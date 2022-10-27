''' Parse the output of the training script for losses and create a plot.
    author: Daniel Nichols
    date: October 2022
'''
# std imports
from argparse import ArgumentParser
from os import PathLike
from typing import Optional, Iterable
from collections import deque
from csv import QUOTE_NONNUMERIC
from string import printable
import json
import math
from os.path import join as path_join

# tpl imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def parse_output(
    results_txt_files: Iterable[PathLike], 
    add_perplexity: bool = True,
    samples_per_step: int = 2
) -> pd.DataFrame:
    ''' Parse the output of a training run.

        Args:
            results_txt_file: paths to the text outputs of training runs
            add_perplexity: calculate perplexity for result if it's not already there
            samples_per_step: how many samples per step were computed
        
        Returns:
            A list of training results in dicts
    '''
    LINE_START_KEY = '{\'loss\':'
    BAD_ESC_STR = ''.join(chr(o) for o in [27, 91, 65])

    results = []
    for text_file in results_txt_files:
        with open(text_file, 'r', encoding='ascii', errors='ignore') as fp:
            prev_lines = deque(3*[None], maxlen=3)
            for line in fp:                
                line = line.strip().replace(BAD_ESC_STR, '')
                if line.startswith(LINE_START_KEY) or LINE_START_KEY in line:
                    obj = json.loads(line.replace('\'', '"'))
                    steps = int( prev_lines[-1].split()[2].split('/')[0] )

                    if add_perplexity and 'perplexity' not in obj and 'loss' in obj:
                        obj['perplexity'] = math.exp(obj['loss'])
                    
                    if 'steps' not in obj:
                        obj['steps'] = steps

                    if 'samples' not in obj:
                        obj['samples'] = obj['steps'] * samples_per_step
                    
                    results.append( obj )
                
                if line.strip() != '':
                    prev_lines.append(line)
    
    return pd.DataFrame( results )


def plot(
    data : pd.DataFrame, 
    output_path : PathLike,
    xcolumn : str = 'samples', 
    ycolumn : str = 'perplexity',
    title : Optional[str] = None
):
    ''' plot the training loss/perplexity curves

        Args:
            data: dataset
            output_path: where to save file
            xcolumn: what column to use for x axis
            ycolumn: what column to use for y axis
    '''
    assert xcolumn in ['samples', 'steps']
    assert ycolumn in ['loss', 'perplexity', 'accuracy']

    plt.clf()
    ax = sns.lineplot(data=data, x=xcolumn, y=ycolumn)
    if title:
        ax.set_title(title)
    plt.savefig(output_path, bbox_inches='tight')


def main():
    parser = ArgumentParser(description='Plot loss and accuracy figures from training results.')
    parser.add_argument('-i', '--input', type=str, nargs='+', required=True, help='training output file')
    parser.add_argument('--output-root', type=str, default='figs', help='director to store figures in')
    parser.add_argument('--samples-per-step', type=int, default=4, help='how many samples are computed per step')
    args = parser.parse_args()

    results = parse_output(args.input, samples_per_step=args.samples_per_step)

    sns.set(font_scale=1.5)
    plot(results, path_join(args.output_root, 'gpt-neo-train-perplexity.png'), xcolumn='samples', ycolumn='perplexity', 
        title='GPT-Neo Training Perplexity')
    plot(results, path_join(args.output_root, 'gpt-neo-train-loss.png'), xcolumn='samples', ycolumn='loss', 
        title='GPT-Neo Training Loss')

    results.to_csv('gpt-neo-training-results.csv', index=False, quoting=QUOTE_NONNUMERIC)

    results = pd.read_csv('gpt-neo-eval-results.csv')
    plot(results, path_join(args.output_root, 'gpt-neo-eval-perplexity.png'), xcolumn='samples', ycolumn='perplexity',
        title='GPT-Neo Evaluation Perplexity')
    plot(results, path_join(args.output_root, 'gpt-neo-eval-loss.png'), xcolumn='samples', ycolumn='loss',
        title='GPT-Neo Evaluation Loss')
    plot(results, path_join(args.output_root, 'gpt-neo-eval-accuracy.png'), xcolumn='samples', ycolumn='accuracy',
        title='GPT-Neo Evaluation Accuracy')



if __name__ == '__main__':
    main()
