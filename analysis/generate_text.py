''' Given one of the models generate some text from a prompt.
    author: Daniel Nichols
    date: October 2022
'''
# std imports
from argparse import ArgumentParser
from os import environ

# tpl imports
from transformers import pipeline


def main():
    parser = ArgumentParser(description='Generate text from a prompt using a LLM.')
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument('--text', type=str, help='Input text to generate new text from.')
    input_group.add_argument('--text-file', type=str, help='File to get text contents from.')
    parser.add_argument('-n', '--num-samples', type=int, default=10, help='How many times to sample a particular input.')
    parser.add_argument('--model', type=str, required=True, help='Huggingface hub model name or path to model.')
    parser.add_argument('--tokenizer', type=str, required=True, help='Tokenizer to use on data.')
    parser.add_argument('--min-len', type=int, default=50, help='Minimum length to generate.')
    parser.add_argument('--max-len', type=int, default=150, help='Maximum length to generate.')
    parser.add_argument('--device', type=int, default=-1, help='Where to run model')
    parser.add_argument('-o', '--output', type=str, default='-', help='Output location. Omit or \'-\' for stdout.')
    args = parser.parse_args()

    # environment setup
    environ['TOKENIZERS_PARALLELISM'] = '0'
    environ['OMP_NUM_THREADS'] = '64'

    # get text data
    prompt = ''
    reprompt = False
    if args.text:
        prompt = args.text
    elif args.text_file:
        with open(args.text_file, 'r', errors='ignore') as fp:
            prompt = fp.read()
    else:
        reprompt = True
        prompt = input('prompt: ')
    
    # create pipeline and generate
    generator = pipeline('text-generation', model=args.model, tokenizer=args.tokenizer, framework='pt', device=args.device)
    
    if reprompt:
        while prompt not in ['q', 'quit', 'exit']:
            prompt = prompt.strip()
            if prompt == '':
                prompt = input('prompt: ')
                continue

            prompts = [prompt] * args.num_samples
            result = generator(prompts, do_sample=True, min_length=args.min_len, max_new_tokens=args.max_len)
            print(result)
            print(result[0]['generated_text'])

            prompt = input('prompt: ')
    else:
        prompts = [prompt] * args.num_samples
        result = generator(prompts, do_sample=True, min_length=args.min_len, max_new_tokens=args.max_len)
    
        # output
        response = result#[0]['generated_text']
        for idx, resp in enumerate(response, start=1):
            gen_text = resp[0]['generated_text']
            if args.output is None or args.output == '-':
                print('Sample {}: \'{}\'\n'.format(idx, gen_text))
            else:
                with open(args.output, 'w') as fp:
                    fp.write('Sample {}: \'{}\'\n'.format(idx, gen_text))


if __name__ == '__main__':
    main()