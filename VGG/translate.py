'''
Translates a source file using a translation model.
'''

import argparse
import theano
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import cPickle as pkl
import numpy
import copy
import pprint
import math
import os
import warnings
import sys
import time
import random
import re
from collections import OrderedDict

from data_iterator import dataIterator

from nmt import (build_sampler, init_params, load_params, init_tparams, 
                 load_dict, init_bn_params)


profile = False


def gen_sample(f_init, f_next, x, options, trng=None, k=1, maxlen=30,
               stochastic=False, argmax=False):
    
    # when testing, stochastic=False
    # k is the beam size we have
    if k > 1:
        assert not stochastic, \
            'Beam search does not support stochastic sampling'

    sample = []
    sample_score = []
    if stochastic:
        sample_score = 0

    live_k = 1 # Top_N = 1
    dead_k = 0

    hyp_samples = [[]] * live_k
    hyp_scores = numpy.zeros(live_k).astype('float32')
    hyp_states = []
    # get initial state of decoder rnn and encoder context
    ret = f_init(x)
    next_state, ctx0 = ret[0], ret[1]
    next_w = -1 * numpy.ones((1,)).astype('int64')  # bos indicator
    next_alpha_past = 0.0 * numpy.ones((1, ctx0.shape[2], ctx0.shape[3])).astype('float32') # start position

    for ii in xrange(maxlen):
        ctx = numpy.tile(ctx0, [live_k, 1, 1, 1])
        inps = [next_w, ctx, next_state, next_alpha_past]
        ret = f_next(*inps)
        next_p, next_w, next_state, next_alpha_past = ret[0], ret[1], ret[2], ret[3]
        if stochastic:
            if argmax:
                nw = next_p[0].argmax()
            else:
                nw = next_w[0]
            sample.append(nw)
            sample_score += next_p[0, nw]
            if nw == 0:
                break
        else:
            cand_scores = hyp_scores[:, None] - numpy.log((next_p))
            cand_flat = cand_scores.flatten()
            ranks_flat = cand_flat.argsort()[:(k-dead_k)]

            voc_size = next_p.shape[1]
            trans_indices = ranks_flat / voc_size
            word_indices = ranks_flat % voc_size
            costs = cand_flat[ranks_flat]

            new_hyp_samples = []
            new_hyp_scores = numpy.zeros(k-dead_k).astype('float32')
            new_hyp_states = []
            new_hyp_alpha_past = []
            for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
                new_hyp_samples.append(hyp_samples[ti]+[wi])
                new_hyp_scores[idx] = copy.copy(costs[idx])
                new_hyp_states.append(copy.copy(next_state[ti]))
                new_hyp_alpha_past.append(copy.copy(next_alpha_past[ti]))

            # check the finished samples
            new_live_k = 0
            hyp_samples = []
            hyp_scores = []
            hyp_states = []
            hyp_alpha_past = []

            for idx in xrange(len(new_hyp_samples)):
                if new_hyp_samples[idx][-1] == 0: # <eol>
                    sample.append(new_hyp_samples[idx])
                    sample_score.append(new_hyp_scores[idx])
                    dead_k += 1
                else:
                    new_live_k += 1
                    hyp_samples.append(new_hyp_samples[idx])
                    hyp_scores.append(new_hyp_scores[idx])
                    hyp_states.append(new_hyp_states[idx])
                    hyp_alpha_past.append(new_hyp_alpha_past[idx])
            hyp_scores = numpy.array(hyp_scores)
            live_k = new_live_k

            if new_live_k < 1:
                break
            if dead_k >= k:
                break

            next_w = numpy.array([w[-1] for w in hyp_samples])
            next_state = numpy.array(hyp_states)
            next_alpha_past = numpy.array(hyp_alpha_past)

    if not stochastic:
        # dump every remaining one
        if live_k > 0:
            for idx in xrange(live_k):
                sample.append(hyp_samples[idx])
                sample_score.append(hyp_scores[idx])

    return sample, sample_score


def main(model, bn_model, dictionary_target, fea, latex, saveto, output, k=5):

    # load model model_options
    with open('%s.pkl' % model, 'rb') as f:
        options = pkl.load(f)

    # load source dictionary and invert
    worddicts = load_dict(dictionary_target)
    worddicts_r = [None] * len(worddicts)
    for kk, vv in worddicts.iteritems():
        worddicts_r[vv] = kk

    valid, valid_uid_list = dataIterator(fea, latex, worddicts,
                         batch_size=1, batch_Imagesize=500000, 
                         maxlen=500, maxImagesize=500000)

    trng = RandomStreams(1234)
    use_noise = theano.shared(numpy.float32(0.))
    

    # allocate model parameters
    params = init_params(options)
    bn_params = init_bn_params(options)
    # load model parameters and set theano shared variables
    params = load_params(model, params)
    bn_params = load_params(bn_model, bn_params)
    tparams = init_tparams(params)
    bn_tparams = init_tparams(bn_params)
    f_init, f_next = build_sampler(tparams, bn_tparams, options, trng, use_noise)

    use_noise.set_value(0.)

    fpp_sample=open(saveto, 'w')
    valid_count_idx=0
    # FIXME: random selection?
    print 'Decoding ... '
    for x, y in valid:
        for xx in x:
            print '%d : %s' % (valid_count_idx+1, valid_uid_list[valid_count_idx])
            xx_pad = numpy.zeros((xx.shape[0],xx.shape[1],xx.shape[2]), dtype='float32') # input_channels * height * width
            xx_pad[:,:, :] = xx / 255.
            stochastic = False
            sample, score = gen_sample(f_init, f_next,
                                       xx_pad[None, :, :, :],
                                       options, trng=trng, k=10,
                                       maxlen=1000,
                                       stochastic=stochastic,
                                       argmax=False)
                        
            if stochastic:
                ss = sample
            else:
                score = score / numpy.array([len(s) for s in sample])
                ss = sample[score.argmin()]

            fpp_sample.write(valid_uid_list[valid_count_idx])
            valid_count_idx=valid_count_idx+1
            for vv in ss:
                if vv == 0: # <eol>
                    break
                fpp_sample.write(' '+worddicts_r[vv])
            fpp_sample.write('\n')
    fpp_sample.close()
    print 'test set decode done'

    os.system('python compute-wer.py ' + saveto + ' ' + latex + ' ' + output)
    fpp=open(output) # %WER 31.63
    stuff=fpp.readlines()
    fpp.close()
    m=re.search('WER (.*)\n',stuff[0])
    valid_per=100. * float(m.group(1))
    m=re.search('ExpRate (.*)\n',stuff[1])
    valid_sacc=100. * float(m.group(1))

    print 'Valid WER: %.2f%%, ExpRate: %.2f%%' % (valid_per,valid_sacc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', type=int, default=5)
    parser.add_argument('model', type=str)
    parser.add_argument('bn_model', type=str)
    parser.add_argument('dictionary_target', type=str)
    parser.add_argument('fea', type=str)
    parser.add_argument('latex', type=str)
    parser.add_argument('saveto', type=str)
    parser.add_argument('output', type=str)

    args = parser.parse_args()

    main(args.model, args.bn_model, args.dictionary_target, args.fea, args.latex,
         args.saveto, args.output, k=args.k)
