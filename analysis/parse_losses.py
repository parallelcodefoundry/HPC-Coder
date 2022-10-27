''' Parse the output of the training script for losses and create a plot.
    author: Daniel Nichols
    date: October 2022
'''
# std imports
from argparse import ArgumentParser
from os import PathLike
from typing import Optional
import json
import math
from os.path import join as path_join

# tpl imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def parse_output(
    results_txt_file: PathLike, 
    add_perplexity: bool = True, 
    log_freq: int = 1, 
    samples_per_step: int = 2
) -> pd.DataFrame:
    ''' Parse the output of a training run.

        Args:
            results_txt_file: path to the text output of a training run
            add_perplexity: calculate perplexity for result if it's not already there
            log_freq: how often values were logged
            samples_per_step: how many samples per step were computed
        
        Returns:
            A list of training results in dicts
    '''
    LINE_START_KEY = '{\'loss\':'

    results = []
    step_counter = 0
    with open(results_txt_file, 'r') as fp:
        for line in fp:
            if line.startswith(LINE_START_KEY):
                obj = json.loads(line.replace('\'', '"'))
                step_counter += log_freq

                if add_perplexity and 'perplexity' not in obj and 'loss' in obj:
                    obj['perplexity'] = math.exp(obj['loss'])
                
                if 'steps' not in obj:
                    obj['steps'] = step_counter

                if 'samples' not in obj:
                    obj['samples'] = obj['steps'] * samples_per_step
                
                results.append( obj )
    
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
    assert ycolumn in ['loss', 'perplexity']

    plt.clf()
    ax = sns.lineplot(data=data, x=xcolumn, y=ycolumn)
    if title:
        ax.set_title(title)
    plt.savefig(output_path)




def main():
    parser = ArgumentParser(description='Plot loss and accuracy figures from training results.')
    parser.add_argument('-i', '--input', type=str, required=True, help='training output file')
    parser.add_argument('--output-root', type=str, default='figs', help='director to store figures in')
    parser.add_argument('--report-frequency', type=int, default=50, help='how often metrics are logged')
    parser.add_argument('--samples-per-step', type=int, default=4, help='how many samples are computed per step')
    args = parser.parse_args()

    results = parse_output(args.input, log_freq=args.report_frequency, samples_per_step=args.samples_per_step)

    sns.set()
    plot(results, path_join(args.output_root, 'gpt-neo-train-perplexity.png'), xcolumn='samples', ycolumn='perplexity', 
        title='GPT-Neo Training Perplexity')
    plot(results, path_join(args.output_root, 'gpt-neo-train-loss.png'), xcolumn='samples', ycolumn='loss', 
        title='GPT-Neo Training Loss')



if __name__ == '__main__':
    main()
