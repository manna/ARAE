from models import load_models, generate
import torch
from torch import nn
from torch.autograd import Variable
import argparse
import numpy as np
import json


def main(args):
    batch_size = 2
    maxlen = 30
    ntokens = len(word2idx)
    reconstruction_loss  = 0
    criterion_ce = nn.CrossEntropyLoss()
    num_lines = 0
    with open(args.output_file, 'w+') as out_file:
        with open(args.input_file, 'r') as read_file:
            for line in read_file.readlines():
                num_lines+=1
                indices =[]
                for w in line.strip('\n').split(' '):
                    try:
                        indices.append(word2idx[w])
                    except KeyError as e:
                        indices.append(word2idx['<oov>'])
                source = Variable(torch.LongTensor(np.array(indices)[np.newaxis]), volatile=True)
                target = Variable(torch.LongTensor(np.array(indices)))
                output = autoencoder.forward(source, [len(indices)], noise=False)
                output = output.squeeze()
                reconstruction_loss+= criterion_ce(output, target).data
                val, op_indices = torch.max(output, 1)

                out_file.write('Orginal: {}'.format(line))
                out_file.write('Reconstructed: {}\n'.format(' '.join(list(map(lambda idx: idx2word[idx], op_indices.data.numpy())))))
            out_file.write('Average Reconstruction loss: {}'.format(reconstruction_loss/num_lines))

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
