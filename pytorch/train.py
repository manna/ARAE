import argparse
from argparse import Namespace
import os
import time
import math
import numpy as np
import random
import sys
import json
import itertools

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from utils import to_gpu, Corpus, batchify, train_ngram_lm, get_ppl
from models import Seq2Seq, MLP_D, MLP_G

parser = argparse.ArgumentParser(description='PyTorch ARAE for Text')

# Path Arguments
parser.add_argument('--kenlm_path', type=str, default='../Data/kenlm',
                    help='path to kenlm directory')
parser.add_argument('--outf', type=str, default='example',
                    help='output directory name')
# Model Arguments
# parser.add_argument('--z_size', type=int, default=100,
#                     help='dimension of random noise z to feed into generator')
parser.add_argument('--temp', type=float, default=1,
                    help='softmax temperature (lower --> more discrete)')
parser.add_argument('--enc_grad_norm', type=bool, default=True,
                    help='norm code gradient from critic->encoder')
parser.add_argument('--gan_toenc', type=float, default=-0.01,
                    help='weight factor passing gradient from gan to encoder')
# Training Arguments
parser.add_argument('--epochs', type=int, default=15,
                    help='maximum number of epochs')
parser.add_argument('--min_epochs', type=int, default=6,
                    help="minimum number of epochs to train for")
parser.add_argument('--no_earlystopping', action='store_true',
                    help="won't use KenLM for early stopping")
parser.add_argument('--patience', type=int, default=5,
                    help="number of language model evaluations without ppl "
                         "improvement to wait before early stopping")
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='batch size')
parser.add_argument('--niters_ae', type=int, default=1,
                    help='number of autoencoder iterations in training')
parser.add_argument('--niters_gan_d', type=int, default=5,
                    help='number of discriminator iterations in training')
parser.add_argument('--niters_gan_g', type=int, default=1,
                    help='number of generator iterations in training')
parser.add_argument('--niters_gan_schedule', type=str, default='2-4-6',
                    help='epoch counts to increase number of GAN training '
                         ' iterations (increment by 1 each time)')
parser.add_argument('--lr_ae', type=float, default=1,
                    help='autoencoder learning rate')
parser.add_argument('--lr_gan_g', type=float, default=5e-05,
                    help='generator learning rate')
parser.add_argument('--lr_gan_d', type=float, default=1e-05,
                    help='critic/discriminator learning rate')
parser.add_argument('--beta1', type=float, default=0.9,
                    help='beta1 for adam. default=0.9')
parser.add_argument('--clip', type=float, default=1,
                    help='gradient clipping, max norm')
parser.add_argument('--gan_clamp', type=float, default=0.01,
                    help='WGAN clamp')

# Evaluation Arguments
parser.add_argument('--sample', action='store_true',
                    help='sample when decoding for generation')
parser.add_argument('--N', type=int, default=5,
                    help='N-gram order for training n-gram language model')
parser.add_argument('--log_interval', type=int, default=200,
                    help='interval to log autoencoder training results')

# Other
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')


ae_parser = argparse.ArgumentParser(description='Autoencoder Arguments Parser')
# Path arguments
ae_parser.add_argument('--data_path', type=str, required=True,
                    help='location of the data corpus')
ae_parser.add_argument('--outf', type=str, default='example',
                    help='output directory name')
# Data Processing Arguments
ae_parser.add_argument('--vocab_size', type=int, default=11000,
                    help='cut vocabulary down to this size '
                         '(most frequently seen words in train)')
ae_parser.add_argument('--maxlen', type=int, default=30,
                    help='maximum sentence length')
ae_parser.add_argument('--lowercase', action='store_true',
                    help='lowercase all text')
# Model Arguments
ae_parser.add_argument('--emsize', type=int, default=300,
                    help='size of word embeddings')
ae_parser.add_argument('--nhidden', type=int, default=300,
                    help='number of hidden units per layer')
ae_parser.add_argument('--nlayers', type=int, default=1,
                    help='number of layers')
ae_parser.add_argument('--noise_radius', type=float, default=0.2,
                    help='stdev of noise for autoencoder (regularizer)')
ae_parser.add_argument('--noise_anneal', type=float, default=0.995,
                    help='anneal noise_radius exponentially by this'
                         'every 100 iterations')
ae_parser.add_argument('--hidden_init', action='store_true',
                    help="initialize decoder hidden state with encoder's")
ae_parser.add_argument('--arch_g', type=str, default='300-300',
                    help='generator architecture (MLP)')
ae_parser.add_argument('--arch_d', type=str, default='300-300',
                    help='critic/discriminator architecture (MLP)')
ae_parser.add_argument('--dropout', type=float, default=0.0,
                help='dropout applied to layers (0 = no dropout)')

mycommands=['ae{}'.format(i) for i in range(5)]
def groupargs(arg,currentarg=[None]):
    if(arg in mycommands):currentarg[0]=arg
    return currentarg[0]

commandlines=[list(args) for cmd,args in itertools.groupby(sys.argv[1:],groupargs)]

args = parser.parse_args(commandlines[0])
autoencoders_args = [ae_parser.parse_args(cl[1:]) for cl in commandlines[1:]]

print '='*8+'ARGUMENTS'+'='*8
print vars(args), '\n'
for ae_args in autoencoders_args:
    print '-'*24
    print vars(ae_args)
print '='*24

# make output directory if it doesn't already exist
for ae_args in autoencoders_args:
    if not os.path.isdir('./output'):
        os.makedirs('./output')
    if not os.path.isdir('./output/{}'.format(ae_args.outf)):
        os.makedirs('./output/{}'.format(ae_args.outf))

# Set the random seed manually for reproducibility.
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, "
              "so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################

# create corpora
for ae_args in autoencoders_args:
    ae_args.corpus = Corpus(
        ae_args.data_path,
        maxlen=ae_args.maxlen,
        vocab_size=ae_args.vocab_size,
        lowercase=ae_args.lowercase)
  
    # dumping vocabulary
    with open('./output/{}/vocab.json'.format(ae_args.outf), 'w') as f:
        json.dump(ae_args.corpus.dictionary.word2idx, f)

    # save arguments
    ntokens = len(ae_args.corpus.dictionary.word2idx)
    print("Vocabulary Size: {}".format(ntokens))
    ae_args.ntokens = ntokens
    
    """
    TODO: Where to write these?
    with open('./output/{}/args.json'.format(args.outf), 'w') as f:
        json.dump(vars(ae_args), f)
    with open("./output/{}/logs.txt".format(args.outf), 'w') as f:
        f.write(str(vars(ae_args)))
        f.write("\n\n")
    """

    eval_batch_size = 10
    ae_args.test_data = batchify(ae_args.corpus.test, eval_batch_size, shuffle=False)
    ae_args.train_data = batchify(ae_args.corpus.train, args.batch_size, shuffle=True)

print("Loaded data!")

###############################################################################
# Build the models
###############################################################################


autoencoders = [Seq2Seq(emsize=ae_args.emsize,
                      nhidden=ae_args.nhidden,
                      ntokens=ae_args.ntokens,
                      nlayers=ae_args.nlayers,
                      noise_radius=ae_args.noise_radius,
                      hidden_init=ae_args.hidden_init,
                      dropout=ae_args.dropout,
                      gpu=args.cuda)
                for ae_args in autoencoders_args]


# gan_gen = MLP_G(ninput=args.z_size, noutput=args.nhidden, layers=args.arch_g)
gan_discs = [MLP_D(ninput=ae_args.nhidden, noutput=1, layers=ae_args.arch_d)
             for ae_args in autoencoders_args]
print('autoencoders', autoencoders)

#print(gan_gen)
#print(gan_disc)

ae_optimizers = [optim.SGD(ae.parameters(), lr=args.lr_ae)
                 for ae, ae_args in zip(autoencoders, autoencoders_args)]
# optimizer_gan_g = optim.Adam(gan_gen.parameters(),
#                             lr=args.lr_gan_g,
#                             betas=(args.beta1, 0.999)) 
gan_d_optimizers = [optim.Adam(gan_disc.parameters(),
                               lr=args.lr_gan_d,
                               betas=(args.beta1, 0.999))
                    for gan_disc, ae_args in zip(gan_discs, autoencoders_args)]

criterion_ce = nn.CrossEntropyLoss()

if args.cuda:
    autoencoders = [ae.cuda() for ae in autoencoders]
    #gan_gen = gan_gen.cuda()
    gan_discs = [gd.cuda() for gd in gan_discs]
    criterion_ce = criterion_ce.cuda()

###############################################################################
# Training code
###############################################################################


def save_model():
    print("Saving models")
    for autoencoder, gan_disc, ae_args in zip(autoencoders, gan_discs, autoencoders_args):
        with open('./output/{}/autoencoder_model.pt'.format(ae_args.outf), 'wb') as f:
            torch.save(autoencoder.state_dict(), f)
        # with open('./output/{}/gan_gen_model.pt'.format(ae_args.outf), 'wb') as f:
            # torch.save(gan_gen.state_dict(), f)
        with open('./output/{}/gan_disc_model.pt'.format(ae_args.outf), 'wb') as f:
            torch.save(gan_disc.state_dict(), f)


def evaluate_autoencoder(autoencoder, ae_args, data_source, epoch):
    # Turn on evaluation mode which disables dropout.
    autoencoder.eval()
    total_loss = 0

    #ntokens = len(ae_args.corpus.dictionary.word2idx)
    ntokens = ae_args.ntokens

    all_accuracies = 0
    bcnt = 0
    for i, batch in enumerate(data_source):
        source, target, lengths = batch
        source = to_gpu(args.cuda, Variable(source, volatile=True))
        target = to_gpu(args.cuda, Variable(target, volatile=True))

        mask = target.gt(0)
        masked_target = target.masked_select(mask)
        # examples x ntokens
        output_mask = mask.unsqueeze(1).expand(mask.size(0), ntokens)

        # output: batch x seq_len x ntokens
        output = autoencoder(source, lengths, noise=True)
        flattened_output = output.view(-1, ntokens)

        masked_output = \
            flattened_output.masked_select(output_mask).view(-1, ntokens)
        total_loss += criterion_ce(masked_output/args.temp, masked_target).data

        # accuracy
        max_vals, max_indices = torch.max(masked_output, 1)
        all_accuracies += \
            torch.mean(max_indices.eq(masked_target).float()).data[0]
        bcnt += 1

        aeoutf = "./output/%s/%d_autoencoder.txt" % (ae_args.outf, epoch)
        with open(aeoutf, "a") as f:
            max_values, max_indices = torch.max(output, 2)
            max_indices = \
                max_indices.view(output.size(0), -1).data.cpu().numpy()
            target = target.view(output.size(0), -1).data.cpu().numpy()
            for t, idx in zip(target, max_indices):
                # real sentence
                chars = " ".join([ae_args.corpus.dictionary.idx2word[x] for x in t])
                f.write(chars)
                f.write("\n")
                # autoencoder output sentence
                chars = " ".join([ae_args.corpus.dictionary.idx2word[x] for x in idx])
                f.write(chars)
                f.write("\n\n")

    return total_loss[0] / len(data_source), all_accuracies/bcnt


def evaluate_generator(noise, epoch):
    gan_gen.eval()
    autoencoder.eval()

    # generate from fixed random noise
    fake_hidden = gan_gen(noise)
    max_indices = \
        autoencoder.generate(fake_hidden, args.maxlen, sample=args.sample)

    with open("./output/%s/%s_generated.txt" % (args.outf, epoch), "w") as f:
        max_indices = max_indices.data.cpu().numpy()
        for idx in max_indices:
            # generated sentence
            words = [corpus.dictionary.idx2word[x] for x in idx]
            # truncate sentences to first occurrence of <eos>
            truncated_sent = []
            for w in words:
                if w != '<eos>':
                    truncated_sent.append(w)
                else:
                    break
            chars = " ".join(truncated_sent)
            f.write(chars)
            f.write("\n")

"""
def train_lm(eval_path, save_path):
    # generate examples
    indices = []
    noise = to_gpu(args.cuda, Variable(torch.ones(100, args.z_size)))
    for i in range(1000):
        noise.data.normal_(0, 1)

        fake_hidden = gan_gen(noise)
        # print ("Calling AE.generate")
        max_indices = autoencoder.generate(fake_hidden, args.maxlen)
        indices.append(max_indices.data.cpu().numpy())

    indices = np.concatenate(indices, axis=0)

    # write generated sentences to text file
    with open(save_path+".txt", "w") as f:
        # laplacian smoothing
        for word in corpus.dictionary.word2idx.keys():
            f.write(word+"\n")
        for idx in indices:
            # generated sentence
            words = [corpus.dictionary.idx2word[x] for x in idx]
            # truncate sentences to first occurrence of <eos>
            truncated_sent = []
            for w in words:
                if w != '<eos>':
                    truncated_sent.append(w)
                else:
                    break
            chars = " ".join(truncated_sent)
            f.write(chars+"\n")

    # train language model on generated examples
    lm = train_ngram_lm(kenlm_path=args.kenlm_path,
                        data_path=save_path+".txt",
                        dedup_data_path=save_path+".uniq.txt",
                        output_path=save_path+".arpa",
                        N=args.N)

    # load sentences to evaluate on
    with open(eval_path, 'r') as f:
        lines = f.readlines()
    sentences = [l.replace('\n', '') for l in lines]
    ppl = get_ppl(lm, sentences)

    return ppl
"""

def train_ae(ae_index, batch, total_loss_ae, start_time, i):
    autoencoder, ae_optimizer = autoencoders[ae_index], ae_optimizers[ae_index]
    ae_args = autoencoders_args[ae_index]    

    autoencoder.train()
    autoencoder.zero_grad()

    source, target, lengths = batch
    source = to_gpu(args.cuda, Variable(source))
    target = to_gpu(args.cuda, Variable(target))

    # Create sentence length mask over padding
    mask = target.gt(0)
    masked_target = target.masked_select(mask)
    # examples x ntokens
    output_mask = mask.unsqueeze(1).expand(mask.size(0), ntokens)

    # output: batch x seq_len x ntokens
    output = autoencoder(source, lengths, noise=True)

    # output_size: batch_size, maxlen, self.ntokens
    flattened_output = output.view(-1, ntokens)

    masked_output = \
        flattened_output.masked_select(output_mask).view(-1, ntokens)
    loss = criterion_ce(masked_output/args.temp, masked_target)
    loss.backward()

    # `clip_grad_norm` to prevent exploding gradient in RNNs / LSTMs
    torch.nn.utils.clip_grad_norm(autoencoder.parameters(), args.clip)
    ae_optimizer.step()

    total_loss_ae += loss.data

    accuracy = None
    if i % args.log_interval == 0 and i > 0:
        # accuracy
        probs = F.softmax(masked_output)
        max_vals, max_indices = torch.max(probs, 1)
        accuracy = torch.mean(max_indices.eq(masked_target).float()).data[0]

        cur_loss = total_loss_ae[0] / args.log_interval
        elapsed = time.time() - start_time
        print('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | '
              'loss {:5.2f} | ppl {:8.2f} | acc {:8.2f}'
              .format(epoch, i, len(ae_args.train_data),
                      elapsed * 1000 / args.log_interval,
                      cur_loss, math.exp(cur_loss), accuracy))

        with open("./output/{}/logs.txt".format(args.outf), 'a') as f:
            f.write('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f} | acc {:8.2f}\n'.
                    format(epoch, i, len(ae_args.train_data),
                           elapsed * 1000 / args.log_interval,
                           cur_loss, math.exp(cur_loss), accuracy))

        total_loss_ae = 0
        start_time = time.time()

    return total_loss_ae, start_time


def train_gan_g():
    gan_gen.train()
    gan_gen.zero_grad()

    noise = to_gpu(args.cuda,
                   Variable(torch.ones(args.batch_size, args.z_size)))
    noise.data.normal_(0, 1)

    fake_hidden = gan_gen(noise)
    errG = gan_disc(fake_hidden)

    # loss / backprop
    errG.backward(one)
    optimizer_gan_g.step()

    return errG


def make_grad_hook(autoencoder):
    def grad_hook(grad):
        # Gradient norm: regularize to be same
        # code_grad_gan * code_grad_ae / norm(code_grad_gan)
        if args.enc_grad_norm:
            gan_norm = torch.norm(grad, 2, 1).detach().data.mean()
            normed_grad = grad * autoencoder.grad_norm / gan_norm
        else:
            normed_grad = grad

        # weight factor and sign flip
        normed_grad *= -math.fabs(args.gan_toenc)
        return normed_grad
    return grad_hook


def train_gan_d(ae_index,  batch):
    autoencoder, optimizer_ae = autoencoders[ae_index], ae_optimizers[ae_index]
    gan_disc, optimizer_gan_d = gan_discs[ae_index], gan_d_optimizers[ae_index]     

    # clamp parameters to a cube
    for p in gan_disc.parameters():
        p.data.clamp_(-args.gan_clamp, args.gan_clamp)

    autoencoder.train()
    autoencoder.zero_grad()
    gan_disc.train()
    gan_disc.zero_grad()

    # positive samples ----------------------------
    # generate real codes
    source, target, lengths = batch
    source = to_gpu(args.cuda, Variable(source))
    target = to_gpu(args.cuda, Variable(target))

    # batch_size x nhidden
    real_hidden = autoencoder(source, lengths, noise=False, encode_only=True)
    real_hidden.register_hook(make_grad_hook(autoencoder))

    # loss / backprop
    errD_real = gan_disc(real_hidden)
    errD_real.backward(one)

    # negative samples ----------------------------i
    # generate fake codes 
    # noise = to_gpu(args.cuda, Variable(torch.ones(args.batch_size, args.z_size)))
    # noise.data.normal_(0, 1)
    
    fake_hiddens = []
    for other_index, other_autoencoder in enumerate(autoencoders):
        if other_index == ae_index: continue
        fake_hidden = other_autoencoder(source, lengths, noise=False, encode_only=True) # TODO: noise=True
        fake_hidden.register_hook(make_grad_hook(other_autoencoder)) # maybe register hook? Not sure. 
        fake_hiddens.append(fake_hidden)        

    # loss / backprop
    # fake_hidden = gan_gen(noise)
    total_errD_fake = None
    errD_fakes = [gan_disc(fake_hidden.detach()) for fake_hidden in fake_hiddens]
    for errD_fake in errD_fakes:
        errD_fake.backward(mone)
        if total_errD_fake is None:
            total_errD_fake = errD_fake
        else:
            total_errD_fake += errD_fake
    # Alernatively, we might prefer: total_errD_fake.backward(mone)

    # `clip_grad_norm` to prvent exploding gradient problem in RNNs / LSTMs
    torch.nn.utils.clip_grad_norm(autoencoder.parameters(), args.clip)

    optimizer_gan_d.step()
    optimizer_ae.step()
    errD = -(errD_real - total_errD_fake)

    return errD, errD_real, total_errD_fake


print("Training...")
with open("./output/{}/logs.txt".format(args.outf), 'a') as f:
    f.write('Training...\n')

# schedule of increasing GAN training loops
if args.niters_gan_schedule != "":
    gan_schedule = [int(x) for x in args.niters_gan_schedule.split("-")]
else:
    gan_schedule = []
niter_gan = 1

# fixed_noise = to_gpu(args.cuda,
#                      Variable(torch.ones(args.batch_size, args.z_size)))
# fixed_noise.data.normal_(0, 1)
one = to_gpu(args.cuda, torch.FloatTensor([1]))
mone = one * -1

best_ppl = None
impatience = 0
all_ppl = []
for epoch in range(1, args.epochs+1):
    # update gan training schedule
    if epoch in gan_schedule:
        niter_gan += 1
        print("GAN training loop schedule increased to {}".format(niter_gan))
        with open("./output/{}/logs.txt".format(args.outf), 'a') as f:
            f.write("GAN training loop schedule increased to {}\n".
                    format(niter_gan))

    for ae_args in autoencoders_args:
        ae_args.epoch_args = Namespace(
            total_loss_ae=0,
            epoch_start_time=time.time(),
            start_time=time.time(),
            niter=0,
            niter_global=1
        )

    # loop through all batches in training data
    while any(ae_args.epoch_args.niter < len(ae_args.train_data) for ae_args in autoencoders_args):

        # train autoencoders ----------------------------
        for i in range(args.niters_ae):
            if ae_args.epoch_args.niter == len(ae_args.train_data):
                break  # end of epoch
            for ae_index, ae_args in enumerate(autoencoders_args):
                ae_args.epoch_args.total_loss_ae, ae_args.epoch_args.start_time = \
                    train_ae(ae_index, ae_args.train_data[ae_args.epoch_args.niter], ae_args.epoch_args.total_loss_ae, ae_args.epoch_args.start_time, ae_args.epoch_args.niter)
            ae_args.epoch_args.niter += 1

        # train gan ----------------------------------
        for k in range(niter_gan):

            # train discriminator/critic
            for i in range(args.niters_gan_d):
                # feed a seen sample within this epoch; good for early training
                for ae_index, ae_args in enumerate(autoencoders_args):
                    errD, errD_real, errD_fake = \
                        train_gan_d(ae_index, ae_args.train_data[random.randint(0, len(ae_args.train_data)-1)])

            # train generator
            # for i in range(args.niters_gan_g):
            #    errG = train_gan_g()

        for ae, ae_args in zip(autoencoders, autoencoders_args):
            ae_args.epoch_args.niter_global += 1
            if ae_args.epoch_args.niter_global % 100 == 0:
                msg = ('[%d/%d][%d/%d] Loss_D: %.8f (Loss_D_real: %.8f'
                    ' Loss_D_fake: %.8f)'  % (epoch, args.epochs, ae_args.epoch_args.niter, len(ae_args.train_data),
                     errD.data[0], errD_real.data[0],
                     errD_fake.data[0]))
                print(msg)
                # with open("./output/{}/logs.txt".format(args.outf), 'a') as f:
                #     f.write(msg)

                # exponentially decaying noise on autoencoder
                ae.noise_radius = ae.noise_radius*ae_args.noise_anneal

                if ae_args.epoch_args.niter_global % 3000 == 0:
                    # evaluate_generator(fixed_noise, "epoch{}_step{}".
                    #                    format(epoch, niter_global))

                    # evaluate with lm
                    if not args.no_earlystopping and epoch > args.min_epochs:
                        # TODO: current implementation requires --no_earlystopping
                        ppl = train_lm(eval_path=os.path.join(ae_args.data_path,
                                                          "test.txt"),
                                   save_path=os.path.abspath("output/{}/"
                                             "epoch{}_step{}_lm_generations".
                                             format(args.outf, epoch,
                                                    ae_args.epoch_args.niter_global)))
                        print("Perplexity {}".format(ppl))
                        all_ppl.append(ppl)
                        print(all_ppl)
                        with open("./output/{}/logs.txt".
                              format(args.outf), 'a') as f:
                            f.write("\n\nPerplexity {}\n".format(ppl))
                            f.write(str(all_ppl)+"\n\n")
                        if best_ppl is None or ppl < best_ppl:
                            impatience = 0
                            best_ppl = ppl
                            print("New best ppl {}\n".format(best_ppl))
                            with open("./output/{}/logs.txt".
                                  format(args.outf), 'a') as f:
                                f.write("New best ppl {}\n".format(best_ppl))
                            save_model()
                        else:
                            impatience += 1
                            # end training
                            if impatience > args.patience:
                                print("Ending training")
                                with open("./output/{}/logs.txt".
                                      format(args.outf), 'a') as f:
                                    f.write("\nEnding Training\n")
                                sys.exit()

    # end of epoch ----------------------------
    # evaluation
    print('-' * 89)
    for ae, ae_args in zip(autoencoders, autoencoders_args):
        test_loss, accuracy = evaluate_autoencoder(ae, ae_args, ae_args.test_data, epoch)
        print('| end of epoch {:3d} | time: {:5.2f}s | test loss {:5.2f} | '
          'test ppl {:5.2f} | acc {:3.3f}'.
            format(epoch, (time.time() - ae_args.epoch_args.epoch_start_time),
                 test_loss, math.exp(test_loss), accuracy))
        print('-' * 89)
        """TODO figure out where to save logs
        with open("./output/{}/logs.txt".format(args.outf), 'a') as f:
            f.write('-' * 89)
            f.write('\n| end of epoch {:3d} | time: {:5.2f}s | test loss {:5.2f} |'
                ' test ppl {:5.2f} | acc {:3.3f}\n'.
                format(epoch, (time.time() - epoch_start_time),
                       test_loss, math.exp(test_loss), accuracy))
            f.write('-' * 89)
            f.write('\n')
        """

    # evaluate_generator(fixed_noise, "end_of_epoch_{}".format(epoch))
    if not args.no_earlystopping and epoch >= args.min_epochs:
        ppl = train_lm(eval_path=os.path.join(args.data_path, "test.txt"),
                       save_path=os.path.abspath("./output/{}/end_of_epoch{}_lm_generations".
                                 format(args.outf, epoch)))
        print("Perplexity {}".format(ppl))
        all_ppl.append(ppl)
        print(all_ppl)
        with open("./output/{}/logs.txt".format(args.outf), 'a') as f:
            f.write("\n\nPerplexity {}\n".format(ppl))
            f.write(str(all_ppl)+"\n\n")
        if best_ppl is None or ppl < best_ppl:
            impatience = 0
            best_ppl = ppl
            print("New best ppl {}\n".format(best_ppl))
            with open("./output/{}/logs.txt".format(args.outf), 'a') as f:
                f.write("New best ppl {}\n".format(best_ppl))
            save_model()
        else:
            impatience += 1
            # end training
            if impatience > args.patience:
                print("Ending training")
                with open("./output/{}/logs.txt".format(args.outf), 'a') as f:
                    f.write("\nEnding Training\n")
                sys.exit()

    # Save the model at the end of every epoch
    if epoch >= args.min_epochs:
        save_model()
    # shuffle between epochs
    for ae_args in autoencoders_args:
        ae_args.train_data = batchify(ae_args.corpus.train, args.batch_size, shuffle=True)
