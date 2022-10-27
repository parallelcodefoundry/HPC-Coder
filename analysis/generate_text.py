''' Given one of the models generate some text from a prompt.
    author: Daniel Nichols
    date: October 2022
'''
# std imports
from argparse import ArgumentParser

# tpl imports
from transformers import pipeline


def main():
    parser = ArgumentParser(description='Generate text from a prompt using a LLM.')
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--text', type=str, help='Input text to generate new text from.')
    input_group.add_argument('--text-file', type=str, help='File to get text contents from.')
    parser.add_argument('--model', type=str, required=True, help='Huggingface hub model name or path to model.')
    parser.add_argument('--min-len', type=int, default=50, help='Minimum length to generate.')
    parser.add_argument('-o', '--output', type=str, default='-', help='Output location. Omit or \'-\' for stdout.')
    args = parser.parse_args()

    # get text data
    prompt = ''
    if args.text:
        prompt = args.text
    else:
        with open(args.text_file, 'r', errors='ignore') as fp:
            prompt = fp.read()
    
    # create pipeline and generate
    generator = pipeline('text-generation', model=args.model)
    result = generator(prompt, do_sample=True, min_length=args.min_len)
    print(result)

    # output
    response = result[0]['generated_text']
    if args.output is None or args.output == '-':
        print(response)
    else:
        with open(args.output, 'w') as fp:
            fp.write(response)


if __name__ == '__main__':
    main()