''' Some summary plots of the repository data.
    author: Daniel Nichols
    date: October 2022
'''
# std imports
from argparse import ArgumentParser
from os import PathLike

# tpl imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def language_histogram(ds: pd.DataFrame, fname: PathLike):
    ''' Plot a histogram of the languages over the repos.
        Args:
            ds: repo metadata dataset
    '''
    plt.clf()
    sns.set()
    hist_fig = sns.histplot(ds, x='language')
    hist_fig.set_title('Repository Main Language Distribution')
    fig = hist_fig.get_figure()
    fig.savefig(fname)


def extensions_histogram(ds: pd.DataFrame, fname: PathLike):
    ''' Plot a histogram of the extensions over the repos.
        Args:
            ds: repo metadata dataset
    '''
    from os.path import splitext
    from load_dataset import get_source_filenames
    ROOT = '/afs/shell.umd.edu/project/bhatele-lab/user/dnicho/code-ml/data/repos'
    fnames = get_source_filenames(ROOT)

    get_extension = lambda x: splitext(x)[-1]
    extensions = list(map(get_extension, fnames))

    plt.clf()
    sns.set()
    hist_fig = sns.histplot(x=extensions)
    hist_fig.set_title('File Type Distribution')
    hist_fig.set_xlabel('File Extension')
    fig = hist_fig.get_figure()
    fig.savefig(fname)


def loc_histogram(ds: pd.DataFrame, fname: PathLike):
    ''' Plot a histogram of the extensions over the repos.
        Args:
            ds: repo metadata dataset
    '''
    from load_dataset import get_source_filenames, get_loc_per_extension, filter_bad_encoding
    ROOT = '/afs/shell.umd.edu/project/bhatele-lab/user/dnicho/code-ml/data/repos'
    fnames = filter_bad_encoding( get_source_filenames(ROOT) )

    loc = get_loc_per_extension(fnames)

    plt.clf()
    sns.set()
    hist_fig = sns.barplot(x=list(loc.keys()), y=list(loc.values()), color='b')
    hist_fig.set_title('Distribution of LOC by File Type')
    hist_fig.set_xlabel('File Extension')
    hist_fig.set_ylabel('LOC')
    fig = hist_fig.get_figure()
    fig.savefig(fname)


def size_histogram(ds: pd.DataFrame, fname: PathLike):
    ''' Plot a histogram of the sizes of the repos.
        Args:
            ds: repo metadata dataset
    '''
    plt.clf()
    sns.set()
    hist_fig = sns.histplot(ds, x='size', stat='count', bins=15, log_scale=(2, False))
    hist_fig.set_title('Repository Size Distribution')
    hist_fig.set_xlabel('Repo Size (KB)')
    fig = hist_fig.get_figure()
    fig.savefig(fname)


def stars_histogram(ds: pd.DataFrame, fname: PathLike):
    ''' Plot a histogram of the stars of the repos.
        Args:
            ds: repo metadata dataset
    '''
    plt.clf()
    sns.set()
    hist_fig = sns.histplot(ds, x='stargazers_count', bins=15, log_scale=(2, False))
    hist_fig.set_title('Repository Stars Distribution')
    hist_fig.set_xlabel('# Stars')
    fig = hist_fig.get_figure()
    fig.savefig(fname)


def watchers_histogram(ds: pd.DataFrame, fname: PathLike):
    ''' Plot a histogram of the watchers of the repos.
        Args:
            ds: repo metadata dataset
    '''
    plt.clf()
    sns.set()
    hist_fig = sns.histplot(ds, x='watchers_count', bins=15, log_scale=(2, False))
    hist_fig.set_title('Repository Watchers Distribution')
    hist_fig.set_xlabel('# Watchers')
    fig = hist_fig.get_figure()
    fig.savefig(fname)


def forks_histogram(ds: pd.DataFrame, fname: PathLike):
    ''' Plot a histogram of the forks of the repos.
        Args:
            ds: repo metadata dataset
    '''
    plt.clf()
    sns.set()
    hist_fig = sns.histplot(ds, x='forks_count', bins=15)
    hist_fig.set_title('Repository Forks Distribution')
    hist_fig.set_xlabel('# Forks')
    fig = hist_fig.get_figure()
    fig.savefig(fname)

def tags_histogram(ds: pd.DataFrame, fname: PathLike):
    ''' Plot a histogram of the tags of the repos.
        Args:
            ds: repo metadata dataset
    '''
    from wordcloud import WordCloud
    def to_list(x):
        return [str(s.strip('\'')) for s in x.strip('][').split(', ') if s != '']

    plt.clf()
    series_as_list = ds['topics'].values.tolist()
    series_as_list = [to_list(x) for x in series_as_list]
    all_tags = [item for sublist in series_as_list for item in sublist]
    
    wc = WordCloud(background_color='white', width=800, height=400).generate(' '.join(all_tags))
    plt.figure(figsize=(20,10))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.savefig(fname, bbox_inches='tight')


def main():
    parser = ArgumentParser(description='script to plot dataset info for repo metadata dataset')
    parser.add_argument('-d', '--dataset', type=str, required=True, help='dataset to use data from')
    parser.add_argument('--languages', type=str, help='create a histogram of languages')
    parser.add_argument('--extensions', type=str, help='create a histogram of language extensions')
    parser.add_argument('--loc', type=str, help='create a histogram of loc per extension')
    parser.add_argument('--sizes', type=str, help='create a histogram of sizes')
    parser.add_argument('--stars', type=str, help='create a histogram of stars')
    parser.add_argument('--watchers', type=str, help='create a histogram of watchers')
    parser.add_argument('--forks', type=str, help='create a histogram of forks')
    parser.add_argument('--tags', type=str, help='create a wordcloud of tags')
    args = parser.parse_args()

    ds = pd.read_csv(args.dataset)

    sns.set_theme()
    if args.languages:
        language_histogram(ds, args.languages)
    if args.extensions:
        extensions_histogram(ds, args.extensions)
    if args.loc:
        loc_histogram(ds, args.loc)
    if args.sizes:
        size_histogram(ds, args.sizes)
    if args.stars:
        stars_histogram(ds, args.stars)
    if args.watchers:
        watchers_histogram(ds, args.watchers)
    if args.forks:
        forks_histogram(ds, args.forks)
    if args.tags:
        tags_histogram(ds, args.tags)



if __name__ == '__main__':
    main()
