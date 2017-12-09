from models import load_models, generate
import torch
from torch.autograd import Variable
import argparse
import numpy as np
import json


def main(args):
    batch_size = 2
    maxlen = 30
    with open(args.output_file, 'w+') as out_file:
        with open(args.input_file, 'r') as read_file:
            for line in read_file.readlines():
                indices =[]
                for w in line.strip('\n').split(' '):
                    try:
                        indices.append(word2idx[w])
                    except KeyError as e:
                        indices.append(word2idx['<oov>'])
                source = Variable(torch.LongTensor(np.array(indices)[np.newaxis]), volatile=True)
                output = autoencoder.forward(source, [len(indices)], noise=False)
                val, op_indices = torch.max(output.squeeze(), 1)
                out_file.write('Orginal: {}'.format(line))
                out_file.write('Reconstructed: {}\n'.format(' '.join(list(map(lambda idx: idx2word[idx], op_indices.data.numpy())))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch experiment')
    parser.add_argument('--load_path', type=str,
                        help='directory to load models from')
    parser.add_argument('--input_file', type=str,
                        help='file containing sentences to modify')
    parser.add_argument('--output_file', type=str,
                        help='file to write output to')
    args = parser.parse_args()
    model_args, idx2word, autoencoder, gan_gen, gan_disc \
        = load_models(args.load_path)
    word2idx = json.load(open("{}/vocab.json".format(args.load_path), "r"))

    main(args)
