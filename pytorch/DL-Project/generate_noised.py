import language_check
import argparse
from nltk import word_tokenize
tool = language_check.LanguageTool('en-US')


def main(args):
    with open(args.output_file, 'w+') as write_file:
        with open(args.input_file, 'r') as read_file:
            for line in read_file.readlines():
                matches = tool.check(line)
                if len(matches) > 0 and len(word_tokenize(line)) <= 10:
                    write_file.write(line)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch experiment')
    parser.add_argument('--input_file', type=str,
                        help='file containing sentences to modify')
    parser.add_argument('--output_file', type=str,
                        help='file to write output to')
    args = parser.parse_args()
    main(args)
