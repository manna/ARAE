import argparse
from models import load_models, generate
import torch
import difflib
import numpy.linalg
import random
import numpy as np
from utils import to_gpu, Corpus, batchify, train_ngram_lm, get_ppl
import os

ENDC = '\033[0m'
BOLD = '\033[1m'


def main(args):
    # Set the random seed manually for reproducibility.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    else:
        print("Note that our pre-trained models require CUDA to evaluate.")

    ###########################################################################
    # Load the models
    ###########################################################################

    model_args, idx2word, autoencoder, gan_gen, gan_disc \
        = load_models(args.load_path)

    ###########################################################################
    # Set Evaluation and save path
    ###########################################################################

    eval_path = os.path.join(args.data_path, "test.txt")
    save_path="output/trained_lm"

    ###########################################################################
    # Generate sentences
    ###########################################################################
    print ("Generating sentences (G->AE)\n")
    noise = torch.ones(args.ngenerations, model_args['z_size'])
    noise.normal_()
    sentences = generate(autoencoder, gan_gen, z=noise,
                         vocab=idx2word, sample=args.sample,
                         maxlen=model_args['maxlen'])

    if not args.noprint:
        print("\nSentence generations:\n")
        for sent in sentences:
            print(sent)
    with open(save_path+".txt", "w") as f:
        f.write("Sentence generations:\n\n")
        for sent in sentences:
            f.write(sent+"\n")

    # train language model on generated examples
    print ("Training N-gram LM\n")
    lm = train_ngram_lm(kenlm_path=args.kenlm_path,
                        data_path=save_path+".txt",
                        output_path=save_path+".arpa",
                        N=args.N)

    # load sentences to evaluate on
    print ("Load Test sentences to evaluate on trained LM\n")
    with open(eval_path, 'r') as f:
        lines = f.readlines()
    sentences = [l.replace('\n', '') for l in lines]
    ppl = get_ppl(lm, sentences)

    print ("Perplexity = " + str(ppl))
    return ppl


def train_lm_real_data(args):
    eval_path = args.data_path + "/test.txt"
    # train language model on generated examples
    print ("Training N-gram LM\n")
    lm = train_ngram_lm(kenlm_path=args.kenlm_path,
                        data_path=args.data_path+"/train.txt",
                        output_path=args.data_path+".arpa",
                        N=args.N)

    # load sentences to evaluate on
    print ("Load Test sentences to evaluate on trained LM\n")
    with open(eval_path, 'r') as f:
        lines = f.readlines()
    sentences = [l.replace('\n',     '') for l in lines]
    ppl = get_ppl(lm, sentences)

    print ("Perplexity = " + str(ppl))
    return ppl


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ARAE for Text Eval')
    parser.add_argument('--load_path', type=str, required=True,
                        help='directory to load models from')
    parser.add_argument('--data_path', type=str, required=True,
                        help='directory to load data from')
    parser.add_argument('--kenlm_path', type=str, default='../Data/kenlm',
                        help='path to kenlm directory')
    parser.add_argument('--ngenerations', type=int, default=10,
                        help='Number of sentences to generate')
    parser.add_argument('--N', type=int, default=5,
                        help='N-Gram')
    parser.add_argument('--R', type=str, required=False,
                        help='Train LM on real data')
    parser.add_argument('--noprint', action='store_true',
                        help='prevents examples from printing')
    parser.add_argument('--real_data', action='store_true',
                        help='Train N-Gram LM on real data')
    parser.add_argument('--sample', action='store_true',
                        help='sample when decoding for generation')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    args = parser.parse_args()
    print(vars(args))
    if args.real_data:
        train_lm_real_data(args)
    else:
        main(args)
