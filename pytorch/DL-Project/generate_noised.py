import language_check
import argparse
import numpy as np
from nltk import word_tokenize
tool = language_check.LanguageTool('en-US')


def main(args):
    count = 3000
    with open(args.output_file, 'w+') as write_file:
        with open(args.input_file, 'r') as read_file:
            for line in read_file.readlines():
                words = word_tokenize(line)
                indices_to_permute = np.arange(len(words))
                np.random.shuffle(indices_to_permute)
                indices_to_permute = sorted(indices_to_permute[:args.swaps])
                permuted_indices = np.random.permutation(indices_to_permute)
                permuted_sentence = []
                idx = 0
                for i in range(len(words)):
                    if i in indices_to_permute:
                        permuted_sentence.append(words[permuted_indices[idx]])
                        idx+=1
                    else:
                        permuted_sentence.append(words[i])
                write_file.write(' '.join(permuted_sentence))
                write_file.write('\n')
                count -= 1
                if count == 0:
                    break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch experiment')
    parser.add_argument('--input_file', type=str,
                        help='file containing sentences to modify')
    parser.add_argument('--output_file', type=str,
                        help='file to write output to')

    parser.add_argument('--swaps', type=int, default=2,
                        help='Number of swaps to perform')
    args = parser.parse_args()
    main(args)
