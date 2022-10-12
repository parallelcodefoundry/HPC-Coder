''' Script to collect github repository metadata.
    author: Daniel Nichols
    date: October 2022
'''
# std imports
from argparse import ArgumentParser
from getpass import getpass
import requests
from csv import QUOTE_NONNUMERIC
from itertools import product
from time import sleep

# tpl imports
import pandas as pd
from alive_progress import alive_it


def query(q, access_token):
    ''' Query github API.
    '''
    BASE_URL = 'https://api.github.com'
    DESIRED_ENTRIES = ['name', 'full_name', 'clone_url', 'html_url', 'created_at',
            'updated_at', 'language', 'size', 'stargazers_count', 'watchers_count',
            'forks_count', 'topics', 'visibility', 'forks', 'open_issues', 'watchers']
    search_url = '{}/search/repositories?q={}'.format(BASE_URL, q)

    fetched = []
    for page in range(1, 35):
        search_url = '{}/search/repositories?q={}&page={}'.format(BASE_URL, q, page)

        try:
            header = {'Authorization': 'token {}'.format(access_token)}
            response = requests.get(search_url, headers=header).json()
        except:
            print('Error fetching from github')
        
        # first try to wait out rate-limiting
        if 'items' not in response:
            sleep(30)
            response = requests.get(search_url, headers=header).json()
        
        if 'items' not in response:
            print(response)
            break
        elif len(response['items']) == 0:
            break

        for item in response['items']:
            vals = { k: item[k] for k in DESIRED_ENTRIES }
            fetched.append(vals)

    return fetched



def collect(tags, languages, min_stars):
    ''' Collect all repo meta-data that match description.
    '''
    api_token = getpass('GitHub API Token: ')

    results = []
    
    bar = alive_it(list(product(tags, languages)))
    for tag, language in bar:
        query_str = 'topic:{} language:{} stars:>={}'.format(tag, language, min_stars)
        query_results = query(query_str, api_token)
        results.extend(query_results)

        # prevent rate-limiting by GitHub API
        #sleep(10)

    return pd.DataFrame(results).drop_duplicates(subset='full_name', ignore_index=True)


def main():
    parser = ArgumentParser(description='Collects repos that meet tag, language, and star requirements.')
    parser.add_argument('--tags', type=str, nargs='+', required=True, help='what tags to look for')
    parser.add_argument('--languages', type=str, nargs='+', default=['c', 'c++'], help='what languages to select repos for')
    parser.add_argument('--min-stars', type=int, default=5, help='minimum number of stars to consider')
    parser.add_argument('-o', '--output', type=str, required=True, help='where to output results')
    args = parser.parse_args()

    df = collect(args.tags, args.languages, args.min_stars)
    
    print('Collected {} repos. Writing to {}.'.format(df.shape[0], args.output))
    df.to_csv(args.output, index=False, quoting=QUOTE_NONNUMERIC)


if __name__ == '__main__':
    main()
