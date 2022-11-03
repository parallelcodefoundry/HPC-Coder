''' Create a dataset for the downstream task of auto-completing openmp pragmas.
    author: Daniel Nichols
    date: November 2022
'''
# std imports
from argparse import ArgumentParser
import json
import re
from typing import Iterable, Optional


PATTERN = r'\#pragma omp parallel for.*'
REG = re.compile(PATTERN, flags=re.MULTILINE)


def strip_comments(text: str) -> str:
    ''' Removes C/C++ style comments from the string.
        Code taken from https://stackoverflow.com/a/241506/3769237

        Args:
            text: input string
        
        Returns:
            The string with C/C++ style comments removed i.e. // /* */
    '''
    def replacer(match):
        s = match.group(0)
        if s.startswith('/'):
            return " " # note: a space and not an empty string
        else:
            return s
    pattern = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE
    )
    return re.sub(pattern, replacer, text)


def get_omp_samples(
    data_sample : dict, 
    pre_loop_token : Optional[str] = '', 
    post_loop_token : Optional[str] = '',
    lines_before : Optional[int] = None,
    chars_before : Optional[int] = None
) -> Iterable[dict]:
    ''' Build a new dataset where each sample is an omp parallel for. Each sample will be a for-loop preceded by 
        `lines_before` or `chars_before` and followed by the openmp pragma.

        Arguments:
            data_sample: a sample object from the dataset. should contain a 'text' column.
            pre_loop_token: token to prepend before loop code.
            post_loop_token: token to append after loop code.
            lines_before: how many lines of context to include before loop.
            chars_before: how many chars of context to include before loop.

        Returns:
            A list of samples with openmp formatted code.
    '''
    assert 'text' in data_sample, 'data_sample must contain the column \'text\'.'
    assert not ((lines_before is not None) and (chars_before is not None)), 'Only one of lines_before and' + \
        ' chars_before can be defined.'

    text = data_sample['text']

    new_samples = []
    for match in REG.finditer(text):
        omp_text = strip_comments( match.group() )
        if '{' in omp_text or '}' in omp_text:
            continue
        
        search_start = match.span()[-1]
        cur_idx = text.find('{', search_start) + 1

        bracket_stack = 1
        failed = False
        while bracket_stack != 0:
            if text[cur_idx] == '{':
                bracket_stack += 1
            elif text[cur_idx] == '}':
                bracket_stack -= 1
            cur_idx += 1

            if cur_idx >= len(text):
                failed = True
                break
        
        if failed:
            # currently cannot handle single statement for loops
            # todo: fix this
            continue

        loop_text = text[search_start : cur_idx].replace('#endif', '').lstrip()
        loop_text = pre_loop_token + loop_text + post_loop_token

        context = ''
        if chars_before is not None:
            pragma_start_idx = match.span()[0]
            offset_idx = max(pragma_start_idx-chars_before, 0)
            context = text[offset_idx : pragma_start_idx]
        elif lines_before is not None:
            raise NotImplementedError('Context by lines not yet supported.')

        new_sample = { k: v for k, v in data_sample.items() if k != 'text' }
        new_sample['omp_pragma_line'] = omp_text
        new_sample['text'] = context + loop_text + ' ' + omp_text
        new_samples.append( new_sample )

    return new_samples


def main():
    parser = ArgumentParser(description='Script that creates the openmp auto-complete dataset.')
    parser.add_argument('-i', '--input', type=str, required=True, help='path to jsonl dataset')
    parser.add_argument('-o', '--output', type=str, required=True, help='path to output dataset from script')
    parser.add_argument('--pre-loop-token', type=str, default='<LOOP START>', help='Token to add before loop.')
    parser.add_argument('--post-loop-token', type=str, default='<LOOP END>', help='Token to add after loop.')
    context_group = parser.add_mutually_exclusive_group(required=True)
    context_group.add_argument('--num-chars-context', type=int, help='how many chars before loop to include')
    context_group.add_argument('--num-lines-context', type=int, help='how many lines before loop to include')
    args = parser.parse_args()

    # process online online, since dataset may be large
    count = 0
    with open(args.input, 'r') as fp_in, open(args.output, 'w') as fp_out:
        for line in fp_in:
            sample = json.loads(line)
            new_samples = get_omp_samples(sample, 
                pre_loop_token=args.pre_loop_token, post_loop_token=args.post_loop_token,
                lines_before=args.num_lines_context, chars_before=args.num_chars_context)

            count += len(new_samples)
            for sample in new_samples:
                json.dump(sample, fp_out, ensure_ascii=False)
                fp_out.write('\n')

    
    print(f'Found {count} total omp decorated for loops.')


if __name__ == '__main__':
    main()
    