from models import load_models, generate
import torch
from torch import nn
from torch.autograd import Variable
from sklearn.metrics.pairwise import cosine_similarity
import argparse
import numpy as np
import json


def get_hidden(sentence):
    indices = []
    for w in sentence.strip('\n').split(' '):
            try:
                indices.append(word2idx[w])
            except KeyError as e:
                indices.append(word2idx['<oov>'])
    source = Variable(torch.LongTensor(np.array(indices)[np.newaxis]), volatile=True)
    output = autoencoder.forward(source, [len(indices)], noise=False, encode_only=True)
    return output.squeeze().data.numpy()


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
    parser.add_argument('--get_hidden', action='store_true')
    parser.add_argument('--sentence', type=str,
                        help='Sentence to get hidden representation for')
    args = parser.parse_args()
    model_args, idx2word, autoencoder, gan_gen, gan_disc \
        = load_models(args.load_path)
    word2idx = json.load(open("{}/vocab.json".format(args.load_path), "r"))

    sentence1 = 'Students raised full-throated slogans against the government.'
    sentence2 = 'Students raised slogans full-throated against the government.'

    # sentence2 = 'The protestors raised full-throated anti-India and pro-liberation slogans'
    output1 = get_hidden(sentence1)
    output2 = get_hidden(sentence2)
    sim = cosine_similarity(output1, output2)
    print(sim[0])
