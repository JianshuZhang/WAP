'''
Build a WAP model with VGG
'''
import theano
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.tensor.signal import pool
from theano.gpuarray.dnn import dnn_batch_normalization_train, dnn_batch_normalization_test
import cPickle as pkl
#import ipdb
import numpy
import copy
import pprint

import os
import warnings
import sys
import math
import time

from collections import OrderedDict

from data_iterator import dataIterator
from optimizers import adadelta, adam, adadelta_weightnoise, adam_weightnoise

profile = False

import random
import re

# push parameters to Theano shared variables
def zipp(params, tparams):
    for kk, vv in params.iteritems():
        tparams[kk].set_value(vv)


# pull parameters from Theano shared variables
def unzip(zipped):
    new_params = OrderedDict()
    for kk, vv in zipped.iteritems():
        new_params[kk] = vv.get_value()
    return new_params


# get the list of parameters: Note that tparams must be OrderedDict
def itemlist(tparams):
    return [vv for kk, vv in tparams.iteritems()]


# dropout
def dropout_layer(state_before, drop_ratio, use_noise, trng):
    proj = tensor.switch(
        use_noise,
        state_before * trng.binomial(state_before.shape, p=1-drop_ratio, n=1,
                                     dtype=state_before.dtype),
        state_before * (1-drop_ratio))
    return proj

def my_mean(state_before, mask, keepdims=False):
    # state_before -- batch * dim * height * width
    #  mask -- batch * height * width
    x = state_before * mask[:, None, :, :]
    x_sum = tensor.sum(x, axis=(0,2,3), keepdims=False)
    x_mean = x_sum / tensor.sum(mask, axis=(0,1,2), keepdims=False)
    return x_mean

def my_var(state_before, x_mean, mask, keepdims=False):
    # state_before -- batch * dim * height * width
    #  mask -- batch * height * width
    #  x_mean -- dim
    x_sub = (state_before - x_mean[None, : ,None, None]) * mask[:, None, :, :]
    x_var = tensor.sum(tensor.sqr(x_sub), axis=(0,2,3), keepdims=False) / tensor.sum(mask, axis=(0,1,2), keepdims=False)
    return x_var

def my_BatchNormalization_mask(state_before, mask, use_noise, x_gamma, x_beta, bn_tparams ,prefix, epsilon = 1e-06):
    # state_before -- batch * dim * height * width
    x_mean = tensor.switch(use_noise,
        my_mean(state_before, mask),
        bn_tparams[prefix+'_bn_mean'])
    x_var = tensor.switch(use_noise,
        my_var(state_before, x_mean, mask),
        bn_tparams[prefix+'_bn_var'])
    after_bn = ((state_before - x_mean[None,:,None,None]) * (x_gamma[None,:,None,None] / tensor.sqrt(x_var[None,:,None,None] + epsilon)) + x_beta[None,:,None,None]) * mask[:, None, :, :]
    mask_sum = tensor.sum(mask, axis=(0,1,2), keepdims=False)
    scale =  mask_sum / (mask_sum-1)
    return after_bn, x_mean, (scale*x_var).astype('float32')

def my_BatchNormalization(state_before, use_noise, x_gamma, x_beta, bn_tparams ,prefix, epsilon = 1e-06):
    # state_before -- batch * dim * height * width
    x_mean = tensor.switch(use_noise,
        tensor.mean(state_before, axis=(0,2,3), keepdims=False),
        bn_tparams[prefix+'_bn_mean'])
    x_var = tensor.switch(use_noise,
        tensor.var(state_before, axis=(0,2,3), keepdims=False),
        bn_tparams[prefix+'_bn_var'])
    after_bn = (state_before - x_mean[None,:,None,None]) * (x_gamma[None,:,None,None] / tensor.sqrt(x_var[None,:,None,None] + epsilon)) + x_beta[None,:,None,None]
    return after_bn, x_mean, x_var

# make prefix-appended name
def _p(pp, name):
    return '%s_%s' % (pp, name)


# initialize Theano shared variables according to the initial parameters
def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams


# load parameters
def load_params(path, params):
    pp = numpy.load(path)
    for kk, vv in params.iteritems():
        if kk not in pp:
            warnings.warn('%s is not in the archive' % kk)
            continue
        params[kk] = pp[kk]

    return params

# layers: 'name': ('parameter initializer', 'feedforward')
layers = {'ff': ('param_init_fflayer', 'fflayer'),
          'gru': ('param_init_gru', 'gru_layer'),
          'gru_cond': ('param_init_gru_cond', 'gru_cond_layer'),
          }


def get_layer(name):
    fns = layers[name]
    return (eval(fns[0]), eval(fns[1]))


# some utilities
def ortho_weight(ndim):
    W = numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype('float32')


def norm_weight(rng, nin, nout):
    fan_in = nin
    fan_out = nout
    W_bound = numpy.sqrt(6. / (fan_in + fan_out))
    W = numpy.asarray(rng.uniform(low=-W_bound, high=W_bound, size=(nin, nout)), dtype=numpy.float32)
    return W.astype('float32')

def conv_norm_weight(rng, nin, nout, kernel_size):
    #rng = numpy.random.RandomState(15992)
    filter_shape = (nout, nin, kernel_size[0], kernel_size[1])
    fan_in = numpy.prod(filter_shape[1:])
    fan_out = filter_shape[0] * numpy.prod(filter_shape[2:])
    W_bound = numpy.sqrt(6. / (fan_in + fan_out))
    W = numpy.asarray(rng.uniform(low=-W_bound, high=W_bound, size=filter_shape), dtype=numpy.float32)
    return W.astype('float32')

def tanh(x):
    return tensor.tanh(x)


def linear(x):
    return x


def concatenate(tensor_list, axis=0):
    """
    Alternative implementation of `theano.tensor.concatenate`.
    This function does exactly the same thing, but contrary to Theano's own
    implementation, the gradient is implemented on the GPU.
    Backpropagating through `theano.tensor.concatenate` yields slowdowns
    because the inverse operation (splitting) needs to be done on the CPU.
    This implementation does not have that problem.
    :usage:
        >>> x, y = theano.tensor.matrices('x', 'y')
        >>> c = concatenate([x, y], axis=1)
    :parameters:
        - tensor_list : list
            list of Theano tensor expressions that should be concatenated.
        - axis : int
            the tensors will be joined along this axis.
    :returns:
        - out : tensor
            the concatenated tensor expression.
    """
    concat_size = sum(tt.shape[axis] for tt in tensor_list)

    output_shape = ()
    for k in range(axis):
        output_shape += (tensor_list[0].shape[k],)
    output_shape += (concat_size,)
    for k in range(axis + 1, tensor_list[0].ndim):
        output_shape += (tensor_list[0].shape[k],)

    out = tensor.zeros(output_shape)
    offset = 0
    for tt in tensor_list:
        indices = ()
        for k in range(axis):
            indices += (slice(None),)
        indices += (slice(offset, offset + tt.shape[axis]),)
        for k in range(axis + 1, tensor_list[0].ndim):
            indices += (slice(None),)

        out = tensor.set_subtensor(out[indices], tt)
        offset += tt.shape[axis]

    return out


# batch preparation
def prepare_data(options, images_x, seqs_y, n_words_src=30000,
                 n_words=30000):
    # x: a list of sentences

    heights_x = [s.shape[1] for s in images_x]
    widths_x = [s.shape[2] for s in images_x]
    lengths_y = [len(s) for s in seqs_y]

    n_samples = len(heights_x)
    max_height_x = numpy.max(heights_x)
    max_width_x = numpy.max(widths_x)
    maxlen_y = numpy.max(lengths_y) + 1

    x = numpy.zeros((n_samples, options['input_channels'], max_height_x, max_width_x)).astype('float32')
    y = numpy.zeros((maxlen_y, n_samples)).astype('int64') # the <eol> must be 0 in the dict !!!
    x_mask = numpy.zeros((n_samples, max_height_x, max_width_x)).astype('float32')
    y_mask = numpy.zeros((maxlen_y, n_samples)).astype('float32')
    for idx, [s_x, s_y] in enumerate(zip(images_x, seqs_y)):
        x[idx, :, :heights_x[idx], :widths_x[idx]] = s_x / 255.
        x_mask[idx, :heights_x[idx], :widths_x[idx]] = 1.
        y[:lengths_y[idx], idx] = s_y
        y_mask[:lengths_y[idx]+1, idx] = 1.

    return x, x_mask, y, y_mask

# conv layer
def param_init_convlayer(rng, options, params, prefix='conv'):
    num_channels = options['input_channels']
    for block in range(0, len(options['dim_ConvBlock'])):
        for level in range(options['layersNum_block'][block]):
            name = 'b{}_l{}'.format(block, level)
            params[_p(prefix, name)] = conv_norm_weight(rng, num_channels, options['dim_ConvBlock'][block], options['kernel_Convenc'])
            bn_gamma  = numpy.asarray(rng.uniform(low=-1.0/math.sqrt(options['dim_ConvBlock'][block]),high=1.0/math.sqrt(options['dim_ConvBlock'][block]),size=options['dim_ConvBlock'][block]),dtype=numpy.float32)
            params[_p(_p(prefix, name), 'bn_gamma')] = bn_gamma[None, :, None, None]
            bn_beta = numpy.zeros((options['dim_ConvBlock'][block])).astype('float32')
            params[_p(_p(prefix, name), 'bn_beta')] = bn_beta[None, :, None, None]
            num_channels = options['dim_ConvBlock'][block]

    return params

def init_bn_params(options):
    params = OrderedDict()
    prefix='conv'
    num_channels = options['input_channels']
    for block in range(0, len(options['dim_ConvBlock'])):
        for level in range(options['layersNum_block'][block]):
            name = 'b{}_l{}'.format(block, level)
            bn_mean = numpy.zeros((options['dim_ConvBlock'][block])).astype('float32')
            params[_p(_p(prefix, name), 'bn_mean')] = bn_mean[None, :, None, None]
            bn_var = numpy.ones((options['dim_ConvBlock'][block])).astype('float32')
            params[_p(_p(prefix, name), 'bn_var')] = bn_var[None, :, None, None]
            num_channels = options['dim_ConvBlock'][block]

    return params

# feedforward layer: affine transformation + point-wise nonlinearity
def param_init_fflayer(rng, options, params, prefix='ff', nin=None, nout=None,
                       ortho=True):
    if nin is None:
        nin = options['dim_proj']
    if nout is None:
        nout = options['dim_proj']
    params[_p(prefix, 'W')] = norm_weight(rng, nin, nout)
    params[_p(prefix, 'b')] = numpy.zeros((nout,)).astype('float32')

    return params


def fflayer(tparams, state_below, options, prefix='rconv',
            activ='lambda x: tensor.tanh(x)', **kwargs):
    return eval(activ)(
        tensor.dot(state_below, tparams[_p(prefix, 'W')]) +
        tparams[_p(prefix, 'b')])


# Conditional GRU layer with Attention
def param_init_gru_cond(rng, options, params, prefix='gru_cond',
                        nin=None, dim=None, dimctx=None, dimatt=None, 
                        nin_nonlin=None, dim_nonlin=None):
    if nin is None:
        nin = options['dim']
    if dim is None:
        dim = options['dim']
    if dimctx is None:
        dimctx = options['dim']
    if dimatt is None:
        dimatt = options['dim_attention']
    if nin_nonlin is None:
        nin_nonlin = nin
    if dim_nonlin is None:
        dim_nonlin = dim

    W = numpy.concatenate([norm_weight(rng, nin, dim),
                           norm_weight(rng, nin, dim)], axis=1)
    params[_p(prefix, 'W')] = W
    params[_p(prefix, 'b')] = numpy.zeros((2 * dim,)).astype('float32')
    U = numpy.concatenate([ortho_weight(dim_nonlin),
                           ortho_weight(dim_nonlin)], axis=1)
    params[_p(prefix, 'U')] = U

    Wx = norm_weight(rng, nin_nonlin, dim_nonlin)
    params[_p(prefix, 'Wx')] = Wx
    Ux = ortho_weight(dim_nonlin)
    params[_p(prefix, 'Ux')] = Ux
    params[_p(prefix, 'bx')] = numpy.zeros((dim_nonlin,)).astype('float32')

    U_nl = numpy.concatenate([ortho_weight(dim_nonlin),
                              ortho_weight(dim_nonlin)], axis=1)
    params[_p(prefix, 'U_nl')] = U_nl
    params[_p(prefix, 'b_nl')] = numpy.zeros((2 * dim_nonlin,)).astype('float32')

    Ux_nl = ortho_weight(dim_nonlin)
    params[_p(prefix, 'Ux_nl')] = Ux_nl
    params[_p(prefix, 'bx_nl')] = numpy.zeros((dim_nonlin,)).astype('float32')

    # context to gru
    Wc = norm_weight(rng, dimctx, dim*2)
    params[_p(prefix, 'Wc')] = Wc

    Wcx = norm_weight(rng, dimctx, dim)
    params[_p(prefix, 'Wcx')] = Wcx

    # attention: combined -> hidden
    W_comb_att = norm_weight(rng, dim, dimatt)
    params[_p(prefix, 'W_comb_att')] = W_comb_att

    # attention: context trans weight
    Wc_att = conv_norm_weight(rng, dimctx, dimatt, [1, 1]).astype('float32')
    params[_p(prefix, 'Wc_att')] = Wc_att

    # attention: context trans bias
    b_att = numpy.zeros((dimatt,)).astype('float32')
    params[_p(prefix, 'b_att')] = b_att

    # attention:
    U_att = norm_weight(rng, dimatt, 1)
    params[_p(prefix, 'U_att')] = U_att
    c_att = numpy.zeros((1,)).astype('float32')
    params[_p(prefix, 'c_tt')] = c_att

    # coverage conv
    params[_p(prefix, 'conv_Q')] = conv_norm_weight(rng, 1, options['dim_coverage'], options['kernel_coverage']).astype('float32')
    params[_p(prefix, 'conv_Uf')] = norm_weight(rng, options['dim_coverage'], dimatt).astype('float32')
    params[_p(prefix, 'conv_b')] = numpy.zeros((dimatt,)).astype('float32')

    return params


def gru_cond_layer(tparams, state_below, options, prefix='gru',
                   mask=None, context=None, one_step=False,
                   init_memory=None, init_state=None, alpha_past=None, 
                   context_mask=None,
                   **kwargs):

    assert context, 'Context must be provided'

    if one_step:
        assert init_state, 'previous state must be provided'

    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    # mask
    if mask is None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    dim = tparams[_p(prefix, 'Wcx')].shape[1]
    dimctx = tparams[_p(prefix, 'Wcx')].shape[0]

    # initial/previous state
    if init_state is None:
        init_state = tensor.alloc(0., n_samples, dim)

    # projected context  traditional -- SeqX * batch * dim
    assert context.ndim == 4, \
        'Context must be 4-d: batch x featuremaps x height x width'

    # Context_mask must be 3-d: batch x height x width

    if alpha_past is None:
        alpha_past = tensor.alloc(0., n_samples, context.shape[2], context.shape[3])

    # before - SeqX * batch * dimctx, now - height * width * batch * dimctx
    pctx_ = theano.tensor.nnet.conv2d(context,tparams[_p(prefix, 'Wc_att')],border_mode='valid') +\
        tparams[_p(prefix, 'b_att')].dimshuffle('x', 0, 'x', 'x')
    pctx_ = pctx_.dimshuffle(2, 3, 0 ,1) # height * width * batch * dimctx

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    # projected x
    state_belowx = tensor.dot(state_below, tparams[_p(prefix, 'Wx')]) +\
        tparams[_p(prefix, 'bx')]
    state_below_ = tensor.dot(state_below, tparams[_p(prefix, 'W')]) +\
        tparams[_p(prefix, 'b')]

    def _step_slice(m_, x_, xx_, h_, ctx_, alpha_, alpha_past_, pctx_, cc_,
                    U, Wc, W_comb_att, U_att, c_tt, Ux, Wcx,
                    U_nl, Ux_nl, b_nl, bx_nl, conv_Q, conv_Uf, conv_b):
        preact1 = tensor.dot(h_, U)
        preact1 += x_
        preact1 = tensor.nnet.sigmoid(preact1)

        r1 = _slice(preact1, 0, dim)
        u1 = _slice(preact1, 1, dim)

        preactx1 = tensor.dot(h_, Ux)
        preactx1 *= r1
        preactx1 += xx_

        h1 = tensor.tanh(preactx1)

        h1 = u1 * h_ + (1. - u1) * h1
        h1 = m_[:, None] * h1 + (1. - m_)[:, None] * h_

        # attention
        pstate_ = tensor.dot(h1, W_comb_att)
        # converage vector
        # alpha_past_ batch * SeqL  !! now alpha_past batch * height * width
        alpha_past__ = alpha_past_[:,None,:,:]
        cover_F = theano.tensor.nnet.conv2d(alpha_past__,conv_Q,border_mode='half') # batch x dim x height x width
        cover_F = cover_F.dimshuffle(2, 3, 0, 1) # height * width * batch * dim
        
        assert cover_F.ndim == 4, \
            'Output of conv must be 4-d: height x width x batch x dimnonlin'
        cover_vector = tensor.dot(cover_F, conv_Uf) + conv_b

        # pctx_ -- height * width * batch * dimctx
        pctx__ = pctx_ + pstate_[None, None, :, :] + cover_vector
        #pctx__ += xc_
        pctx__ = tensor.tanh(pctx__)
        alpha = tensor.dot(pctx__, U_att)+c_tt
        alpha = alpha.reshape([alpha.shape[0], alpha.shape[1], alpha.shape[2]]) # height * width * batch
        alpha = tensor.exp(alpha)
        if context_mask:
            alpha = alpha * context_mask.dimshuffle(1, 2, 0)
        alpha = alpha / alpha.sum(axis=(0,1), keepdims=True)
        alpha_past = alpha_past_ + alpha.dimshuffle(2, 0, 1) # batch * height * width
        ctx_ = (cc_ * alpha.dimshuffle(2, 0, 1)[:,None,:,:]).sum(axis=(2,3))  # current context

        preact2 = tensor.dot(h1, U_nl)+b_nl
        preact2 += tensor.dot(ctx_, Wc)
        preact2 = tensor.nnet.sigmoid(preact2)

        r2 = _slice(preact2, 0, dim)
        u2 = _slice(preact2, 1, dim)

        preactx2 = tensor.dot(h1, Ux_nl)+bx_nl
        preactx2 *= r2
        preactx2 += tensor.dot(ctx_, Wcx)

        h2 = tensor.tanh(preactx2)

        h2 = u2 * h1 + (1. - u2) * h2
        h2 = m_[:, None] * h2 + (1. - m_)[:, None] * h1

        return h2, ctx_, alpha.dimshuffle(2, 0, 1), alpha_past  # pstate_, preact, preactx, r, u

    seqs = [mask, state_below_, state_belowx]
    #seqs = [mask, state_below_, state_belowx, state_belowc]
    _step = _step_slice

    shared_vars = [tparams[_p(prefix, 'U')],
                   tparams[_p(prefix, 'Wc')],
                   tparams[_p(prefix, 'W_comb_att')],
                   tparams[_p(prefix, 'U_att')],
                   tparams[_p(prefix, 'c_tt')],
                   tparams[_p(prefix, 'Ux')],
                   tparams[_p(prefix, 'Wcx')],
                   tparams[_p(prefix, 'U_nl')],
                   tparams[_p(prefix, 'Ux_nl')],
                   tparams[_p(prefix, 'b_nl')],
                   tparams[_p(prefix, 'bx_nl')],
                   tparams[_p(prefix, 'conv_Q')],
                   tparams[_p(prefix, 'conv_Uf')],
                   tparams[_p(prefix, 'conv_b')]]

    if one_step:
        rval = _step(*(seqs + [init_state, None, None, alpha_past, pctx_, context] +
                       shared_vars))
    else:
        rval, updates = theano.scan(_step,
                                    sequences=seqs,
                                    outputs_info=[init_state,
                                                  tensor.alloc(0., n_samples,
                                                               context.shape[1]),
                                                  tensor.alloc(0., n_samples,
                                                               context.shape[2], context.shape[3]),
                                                  tensor.alloc(0., n_samples,
                                                               context.shape[2], context.shape[3])],
                                    non_sequences=[pctx_, context]+shared_vars,
                                    name=_p(prefix, '_layers'),
                                    n_steps=nsteps,
                                    profile=profile,
                                    strict=True)
    return rval


# initialize all parameters
def init_params(options):
    rng = numpy.random.RandomState(int(time.time()))
    params = OrderedDict()

    # embedding
    params['Wemb_dec'] = norm_weight(rng, options['dim_target'], options['dim_word'])

    num_channels = options['input_channels']
    for block in range(0, len(options['dim_ConvBlock'])):
        for level in range(options['layersNum_block'][block]):
            name = 'b{}_l{}'.format(block, level)
            params[_p('conv', name)] = conv_norm_weight(rng, num_channels, options['dim_ConvBlock'][block], options['kernel_Convenc'])
            bn_gamma = numpy.asarray(rng.uniform(low=-1.0/math.sqrt(options['dim_ConvBlock'][block]),high=1.0/math.sqrt(options['dim_ConvBlock'][block]),size=options['dim_ConvBlock'][block]),dtype=numpy.float32)
            params[_p(_p('conv', name), 'bn_gamma')] = bn_gamma[None, :, None, None]
            bn_beta = numpy.zeros((options['dim_ConvBlock'][block])).astype('float32')
            params[_p(_p('conv', name), 'bn_beta')] = bn_beta[None, :, None, None]
            num_channels = options['dim_ConvBlock'][block]

    ctxdim = options['dim_ConvBlock'][-1]

    # init_state, init_cell
    params = get_layer('ff')[0](rng, options, params, prefix='ff_state',
                                nin=ctxdim, nout=options['dim_dec'])
    # decoder
    params = get_layer(options['decoder'])[0](rng, options, params,
                                              prefix='decoder',
                                              nin=options['dim_word'],
                                              dim=options['dim_dec'],
                                              dimctx=ctxdim,
                                              dimatt=options['dim_attention'])
    # readout
    params = get_layer('ff')[0](rng, options, params, prefix='ff_logit_gru',
                                nin=options['dim_dec'], nout=options['dim_word'],
                                ortho=False)
    params = get_layer('ff')[0](rng, options, params, prefix='ff_logit_prev',
                                nin=options['dim_word'],
                                nout=options['dim_word'], ortho=False)
    params = get_layer('ff')[0](rng, options, params, prefix='ff_logit_ctx',
                                nin=ctxdim, nout=options['dim_word'],
                                ortho=False)
    params = get_layer('ff')[0](rng, options, params, prefix='ff_logit',
                                nin=options['dim_word']/2,
                                nout=options['dim_target'])

    return params


# build a training model
def build_model(tparams, bn_tparams, options):
    opt_ret = dict()

    trng = RandomStreams(1234)
    use_noise = theano.shared(numpy.float32(0.))

    # description string: #words x #samples
    x = tensor.tensor4('x', dtype='float32') #batch * input_channels * height * width
    x_mask_original = tensor.tensor3('x_mask_original', dtype='float32') # batch * height * width
    y = tensor.matrix('y', dtype='int64')
    y_mask = tensor.matrix('y_mask', dtype='float32')
    x_mask = x_mask_original

    #n_timesteps = x.shape[0]
    n_timesteps_trg = y.shape[0]
    n_samples = x.shape[0]

    oup = x
    for block in range(0,len(options['dim_ConvBlock'])):
        for level in range(options['layersNum_block'][block]):
            name = 'b{}_l{}'.format(block, level)
            conv_filter = tparams[_p('conv', name)]
            oup=theano.tensor.nnet.conv2d(oup, conv_filter, border_mode='half', subsample=(1,1))
            bn_gamma = tparams[_p(_p('conv', name), 'bn_gamma')]
            bn_beta = tparams[_p(_p('conv', name), 'bn_beta')]
            bn_mean = bn_tparams[_p(_p('conv', name), 'bn_mean')]
            bn_var = bn_tparams[_p(_p('conv', name), 'bn_var')]
            oup, s_mean, s_invstd, opt_ret[_p(_p('conv', name), 'bn_mean')], opt_ret[_p(_p('conv', name), 'bn_var')] = dnn_batch_normalization_train(oup, bn_gamma, bn_beta, mode='spatial', epsilon=0.0001, running_mean=bn_mean, running_var=bn_var)
            #oup, opt_ret[_p(_p('conv', name), 'bn_mean')], opt_ret[_p(_p('conv', name), 'bn_var')] = my_BatchNormalization_mask(oup, x_mask, use_noise, tparams[_p(_p('conv', name), 'bn_gamma')], tparams[_p(_p('conv', name), 'bn_beta')], bn_tparams, _p('conv', name))
            oup=tensor.nnet.relu(oup)
            if options['use_dropout']:
                if name in ['b3_l0', 'b3_l1', 'b3_l2']:
                    oup=dropout_layer(oup, 0.2, use_noise,trng)
        height_pad = tensor.alloc(-1., oup.shape[0], oup.shape[1], 1, oup.shape[3])
        oup = concatenate([oup, height_pad], axis=2)
        width_pad = tensor.alloc(-1., oup.shape[0], oup.shape[1], oup.shape[2], 1)
        oup = concatenate([oup, width_pad], axis=3)
        oup=pool.pool_2d(oup,(2,2),ignore_border=True,stride=(2,2),pad=(0,0),mode='max')
        x_mask=x_mask[:,0::2,0::2]

    # oup batch * featuremaps * height * width
    ctx = oup

    # mean of the context (across time) will be used to initialize decoder rnn
    ctx_mean = (ctx * x_mask[:, None, :, :]).sum(axis=(2,3)) / x_mask.sum(axis=(1,2))[:, None]

    # or you can use the last state of forward + backward encoder rnns
    # ctx_mean = concatenate([proj[0][-1], projr[0][-1]], axis=proj[0].ndim-2)

    # initial decoder state
    init_state = get_layer('ff')[1](tparams, ctx_mean, options,
                                    prefix='ff_state', activ='tanh')

    # word embedding (target), we will shift the target sequence one time step
    # to the right. This is done because of the bi-gram connections in the
    # readout and decoder rnn. The first target will be all zeros and we will
    # not condition on the last output.
    emb = tparams['Wemb_dec'][y.flatten()]
    emb = emb.reshape([n_timesteps_trg, n_samples, options['dim_word']])
    emb_shifted = tensor.zeros_like(emb) # the 0 idx is <eos>!!
    emb_shifted = tensor.set_subtensor(emb_shifted[1:], emb[:-1])
    emb = emb_shifted

    # decoder - pass through the decoder conditional gru with attention
    proj = get_layer(options['decoder'])[1](tparams, emb, options,
                                            prefix='decoder',
                                            mask=y_mask, context=ctx,
                                            context_mask=x_mask,
                                            one_step=False,
                                            init_state=init_state)
    # hidden states of the decoder gru
    proj_h = proj[0]

    # weighted averages of context, generated by attention module
    ctxs = proj[1]

    # weights (alignment matrix)
    opt_ret['dec_alphas'] = proj[2]

    # compute word probabilities
    logit_gru = get_layer('ff')[1](tparams, proj_h, options,
                                    prefix='ff_logit_gru', activ='linear')
    logit_prev = get_layer('ff')[1](tparams, emb, options,
                                    prefix='ff_logit_prev', activ='linear')
    logit_ctx = get_layer('ff')[1](tparams, ctxs, options,
                                   prefix='ff_logit_ctx', activ='linear')
    logit = logit_gru+logit_prev+logit_ctx

    # maxout 2
    # maxout layer
    shape = logit.shape
    shape2 = tensor.cast(shape[2] / 2, 'int64')
    shape3 = tensor.cast(2, 'int64')
    logit = logit.reshape([shape[0],shape[1], shape2, shape3]) # seq*batch*256 -> seq*batch*128*2
    logit=logit.max(3) # seq*batch*128

    if options['use_dropout']:
        logit = dropout_layer(logit, 0.2, use_noise, trng)
    logit = get_layer('ff')[1](tparams, logit, options, 
                               prefix='ff_logit', activ='linear')
    logit_shp = logit.shape
    probs = tensor.nnet.softmax(logit.reshape([logit_shp[0]*logit_shp[1],
                                               logit_shp[2]]))

    # cost
    cost = tensor.nnet.categorical_crossentropy(probs, y.flatten()) # x is a vector,each value is a 1-of-N position 
    cost = cost.reshape([y.shape[0], y.shape[1]])
    cost = (cost * y_mask).sum(0) 

    return trng, use_noise, x, x_mask_original, y, y_mask, opt_ret, cost


# build a sampler
def build_sampler(tparams, bn_tparams, options, trng, use_noise):
    x = tensor.tensor4('x', dtype='float32')
    opt_ret = dict()

    n_samples = x.shape[0]

    oup = x # batch * input_channels * height * width
    for block in range(0,len(options['dim_ConvBlock'])):
        for level in range(options['layersNum_block'][block]):
            name = 'b{}_l{}'.format(block, level)
            conv_filter = tparams[_p('conv', name)]
            oup=theano.tensor.nnet.conv2d(oup, conv_filter, border_mode='half', subsample=(1,1)) 
            bn_gamma = tparams[_p(_p('conv', name), 'bn_gamma')]
            bn_beta = tparams[_p(_p('conv', name), 'bn_beta')]
            bn_mean = bn_tparams[_p(_p('conv', name), 'bn_mean')]
            bn_var = bn_tparams[_p(_p('conv', name), 'bn_var')]
            oup = dnn_batch_normalization_test(oup, bn_gamma, bn_beta, bn_mean, bn_var, mode='spatial', epsilon=0.0001)
            #oup, opt_ret[_p(_p('conv', name), 'bn_mean')], opt_ret[_p(_p('conv', name), 'bn_var')] = my_BatchNormalization(oup, use_noise, tparams[_p(_p('conv', name), 'bn_gamma')], tparams[_p(_p('conv', name), 'bn_beta')], bn_tparams, _p('conv', name))
            oup=tensor.nnet.relu(oup)
            if options['use_dropout']:
                if name in ['b3_l0', 'b3_l1', 'b3_l2']:
                    oup=dropout_layer(oup, 0.2, use_noise,trng)
        height_pad = tensor.alloc(-1., oup.shape[0], oup.shape[1], 1, oup.shape[3])
        oup = concatenate([oup, height_pad], axis=2)
        width_pad = tensor.alloc(-1., oup.shape[0], oup.shape[1], oup.shape[2], 1)
        oup = concatenate([oup, width_pad], axis=3)
        oup=pool.pool_2d(oup,(2,2),ignore_border=True,stride=(2,2),pad=(0,0),mode='max')


    ctx = oup # batch * featuremaps * height * width
    # get the input for decoder rnn initializer mlp
    ctx_mean = ctx.mean(axis=(2,3))
    # ctx_mean = concatenate([proj[0][-1],projr[0][-1]], axis=proj[0].ndim-2)
    init_state = get_layer('ff')[1](tparams, ctx_mean, options,
                                    prefix='ff_state', activ='tanh')

    print 'Building f_init...',
    outs = [init_state, ctx]
    f_init = theano.function([x], outs, name='f_init', profile=profile,allow_input_downcast=True)
    print 'Done'

    # x: 1 x 1
    y = tensor.vector('y_sampler', dtype='int64')
    init_state = tensor.matrix('init_state', dtype='float32')
    alpha_past = tensor.tensor3('alpha_past', dtype='float32')

    # if it's the first word, emb should be all zero and it is indicated by -1
    emb = tensor.switch(y[:, None] < 0,
                        tensor.alloc(0., 1, tparams['Wemb_dec'].shape[1]),
                        tparams['Wemb_dec'][y])

    # apply one step of conditional gru with attention
    proj = get_layer(options['decoder'])[1](tparams, emb, options,
                                            prefix='decoder',
                                            mask=None, context=ctx, 
                                            one_step=True,
                                            init_state=init_state, alpha_past = alpha_past)
    # get the next hidden state
    next_state = proj[0]

    # get the weighted averages of context for this target word y
    ctxs = proj[1]
    next_alpha_past = proj[3]

    logit_gru = get_layer('ff')[1](tparams, next_state, options,
                                    prefix='ff_logit_gru', activ='linear')
    logit_prev = get_layer('ff')[1](tparams, emb, options,
                                    prefix='ff_logit_prev', activ='linear')
    logit_ctx = get_layer('ff')[1](tparams, ctxs, options,
                                   prefix='ff_logit_ctx', activ='linear')
    logit = logit_gru+logit_prev+logit_ctx

    # maxout layer
    shape = logit.shape
    shape1 = tensor.cast(shape[1] / 2, 'int64')
    shape2 = tensor.cast(2, 'int64')
    logit = logit.reshape([shape[0], shape1, shape2]) # batch*256 -> batch*128*2
    logit=logit.max(2) # batch*500

    if options['use_dropout']:
        logit = dropout_layer(logit, 0.2, use_noise, trng)
    logit = get_layer('ff')[1](tparams, logit, options,
                               prefix='ff_logit', activ='linear')

    # compute the softmax probability
    next_probs = tensor.nnet.softmax(logit)

    # sample from softmax distribution to get the sample
    next_sample = trng.multinomial(pvals=next_probs).argmax(1)

    # compile a function to do the whole thing above, next word probability,
    # sampled word for the next target, next hidden state to be used
    print 'Building f_next...',
    inps = [y, ctx, init_state, alpha_past]
    outs = [next_probs, next_sample, next_state, next_alpha_past]
    f_next = theano.function(inps, outs, name='f_next', profile=profile,allow_input_downcast=True)
    print 'Done'

    return f_init, f_next


# generate sample, either with stochastic sampling or beam search. Note that,
# this function iteratively calls f_init and f_next functions.
def gen_sample(tparams, f_init, f_next, x, options, trng=None, k=1, maxlen=30,
               stochastic=True, argmax=False):

    # k is the beam size we have
    if k > 1:
        assert not stochastic, \
            'Beam search does not support stochastic sampling'

    sample = []
    sample_score = []
    if stochastic:
        sample_score = 0

    live_k = 1
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
            cand_scores = hyp_scores[:, None] - numpy.log(next_p)
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


# calculate the log probablities on a given corpus using translation model
def pred_probs(f_log_probs, prepare_data, options, iterator, verbose=False):
    probs = []

    n_done = 0

    for x, y in iterator:
        n_done += len(x)

        x, x_mask, y, y_mask = prepare_data(options, x, y)

        pprobs = f_log_probs(x, x_mask, y, y_mask)
        for pp in pprobs:
            probs.append(pp)

        if numpy.isnan(numpy.mean(probs)):
            #ipdb.set_trace()
            print 'probs nan'

        if verbose:
            print >>sys.stderr, '%d samples computed' % (n_done)

    return numpy.array(probs)


def load_dict(dictFile):
    fp=open(dictFile)
    stuff=fp.readlines()
    fp.close()
    lexicon={}
    for l in stuff:
        w=l.strip().split()
        lexicon[w[0]]=int(w[1])

    print 'total words/phones',len(lexicon)
    return lexicon




def train(dim_word=100,  # word vector dimensionality
          dim_dec=1000,
          dim_attention=512,
          dim_coverage=512,
          kernel_coverage=[5,5],
          kernel_Convenc=[3,1],
          dim_ConvBlock=[32,64,64,128],
          layersNum_block=[4,4,4,4],
          encoder='gru',
          decoder='gru_cond',
          patience=4,  # early stopping patience
          max_epochs=5000,
          finish_after=10000000,  # finish after this many updates
          dispFreq=100,
          decay_c=0.,  # L2 regularization penalty
          alpha_c=0.,  # alignment regularization
          clip_c=-1.,  # gradient clipping threshold
          lrate=1e-8,  # learning rate
          dim_target=62,  # source vocabulary size
          input_channels=123,  # target vocabulary size
          maxlen=100,  # maximum length of the description
          maxImagesize=1, # maximum size of the input image
          optimizer='rmsprop',
          batch_Imagesize=16,
          valid_batch_Imagesize=16,
          batch_size=16,
          valid_batch_size=16,
          saveto='model.npz',
          bn_saveto='bn_model.npz',
          validFreq=1000,
          saveFreq=1000,   # save the parameters after every saveFreq updates
          sampleFreq=100,   # generate some samples after every sampleFreq
          datasets=['feature.pkl',
                    'label.txt'],
          valid_datasets=['feature_valid.pkl', 
                          'label_valid.txt'],
          dictionaries=['lexicon.txt'],
          valid_output=['decode.txt'],
          valid_result=['result.txt'],
          use_dropout=False,
          reload_=False):

    # Model options
    model_options = locals().copy()

    # load dictionaries and invert them

    worddicts = load_dict(dictionaries[0])
    worddicts_r = [None] * len(worddicts)
    for kk, vv in worddicts.iteritems():
        worddicts_r[vv] = kk

    # reload options
    if reload_ and os.path.exists(saveto):
        with open('%s.pkl' % saveto, 'rb') as f:
            models_options = pkl.load(f)

    print 'Loading data'

    train,train_uid_list = dataIterator(datasets[0], datasets[1],
                         worddicts,
                         batch_size=batch_size, batch_Imagesize=batch_Imagesize,maxlen=maxlen,maxImagesize=maxImagesize)
    valid,valid_uid_list = dataIterator(valid_datasets[0], valid_datasets[1],
                         worddicts,
                         batch_size=valid_batch_size, batch_Imagesize=valid_batch_Imagesize,maxlen=maxlen,maxImagesize=maxImagesize)

    print 'Building model'
    params = init_params(model_options)
    bn_params = init_bn_params(model_options)
    # reload parameters
    if reload_ and os.path.exists(saveto):
        params = load_params(saveto, params)
        bn_params = load_params(bn_saveto, bn_params)

    tparams = init_tparams(params)
    bn_tparams = init_tparams(bn_params)

    trng, use_noise, \
        x, x_mask, y, y_mask, \
        opt_ret, \
        cost = \
        build_model(tparams, bn_tparams, model_options)
    inps = [x, x_mask, y, y_mask]

    print 'Buliding sampler'
    f_init, f_next = build_sampler(tparams, bn_tparams, model_options, trng, use_noise)

    # before any regularizer
    print 'Building f_log_probs...',
    f_log_probs = theano.function(inps, cost, profile=profile)
    print 'Done'

    cost = cost.mean()

    # apply L2 regularization on weights
    if decay_c > 0.:
        decay_c = theano.shared(numpy.float32(decay_c), name='decay_c')
        weight_decay = 0.
        for kk, vv in tparams.iteritems():
            tmp = kk.split('_')
            if tmp[-2] != 'bn':
                weight_decay += (vv ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay

    # regularize the alpha weights
    if alpha_c > 0. and not model_options['decoder'].endswith('simple'):
        alpha_c = theano.shared(numpy.float32(alpha_c), name='alpha_c')
        alpha_reg = alpha_c * (
            (tensor.cast(y_mask.sum(0)//x_mask.sum(0), 'float32')[:, None] -
             opt_ret['dec_alphas'].sum(0))**2).sum(1).mean()
        cost += alpha_reg

    # after all regularizers - compile the computational graph for cost
    print 'Building f_cost...',
    f_cost = theano.function(inps, cost, profile=profile)
    print 'Done'

    print 'Computing gradient...',
    grads = tensor.grad(cost, wrt=itemlist(tparams))
    print 'Done'

    # apply gradient clipping here
    if clip_c > 0.:
        g2 = 0.
        for g in grads:
            g2 += (g**2).sum()
        new_grads = []
        for g in grads:
            new_grads.append(tensor.switch(g2 > (clip_c**2),
                                           g / tensor.sqrt(g2) * clip_c,
                                           g))
        grads = new_grads

    # compile the optimizer, the actual computational graph is compiled here
    lr = tensor.scalar(name='lr')
    print 'Building optimizers...',
    f_grad_shared, f_update = eval(optimizer)(lr, tparams, bn_tparams, opt_ret, grads, inps, cost)
    print 'Done'

    
    
    # print model parameters
    print "Model params:\n{0}".format(
            pprint.pformat(sorted([p for p in params])))
    # end



    print 'Optimization'

    history_errs = []
    # reload history
    if reload_ and os.path.exists(saveto):
        history_errs = list(numpy.load(saveto)['history_errs'])
    best_p = None
    best_bn_p = None
    bad_count = 0

    if validFreq == -1:
        validFreq = len(train)
    if saveFreq == -1:
        saveFreq = len(train)
    if sampleFreq == -1:
        sampleFreq = len(train)

    uidx = 0
    estop = False
    halfLrFlag = 0
    bad_counter = 0
    ud_s = 0
    ud_epoch = 0
    cost_s = 0.
    for eidx in xrange(max_epochs):
        n_samples = 0

        ud_epoch = time.time()
        random.shuffle(train) # shuffle data

        for x, y in train:
            n_samples += len(x)
            uidx += 1
            use_noise.set_value(1.)

            ud_start = time.time()

            x, x_mask, y, y_mask = prepare_data(model_options, x, y)

            if x is None:
                print 'Minibatch with zero sample under length ', maxlen
                uidx -= 1
                continue

            # compute cost, grads and copy grads to shared variables
            cost = f_grad_shared(x, x_mask, y, y_mask)
            cost_s += cost
            # do the update on parameters
            f_update(lrate)

            ud = time.time() - ud_start
            ud_s += ud
            

            # check for bad numbers, usually we remove non-finite elements
            # and continue training - but not done here
            if numpy.isnan(cost) or numpy.isinf(cost):
                print 'NaN detected'
                return 1., 1., 1.

            # verbose
            if numpy.mod(uidx, dispFreq) == 0:
                ud_s /= 60.
                cost_s /= dispFreq
                print 'Epoch ', eidx, 'Update ', uidx, 'Cost ', cost_s, 'UD ', ud_s, 'lrate ',lrate, 'bad_counter', bad_counter
                ud_s = 0
                cost_s = 0.

            # save the best model so far
            if numpy.mod(uidx, saveFreq) == 0:
                print 'Saving...',

                if best_p is not None:
                    params = best_p
                    bn_params = best_bn_p
                else:
                    params = unzip(tparams)
                    bn_params = unzip(bn_tparams)
                numpy.savez(saveto, history_errs=history_errs, **params)
                numpy.savez(bn_saveto, history_errs=history_errs, **bn_params)
                pkl.dump(model_options, open('%s.pkl' % saveto, 'wb'))
                print 'Done'

            # generate some samples with the model and display them
            if numpy.mod(uidx, sampleFreq) == 0:
                # FIXME: random selection?
                use_noise.set_value(0.)
                fpp_sample=open(valid_output[0],'w')
                valid_count_idx=0
                # FIXME: random selection?
                for x,y in valid:
                    for xx in x:
                        xx_pad = numpy.zeros((xx.shape[0],xx.shape[1],xx.shape[2]), dtype='float32') # input_channels * height * width
                        xx_pad[:,:, :] = xx / 255.
                        stochastic = False
                        sample, score = gen_sample(tparams, f_init, f_next,
                                                   xx_pad[None, :, :, :],
                                                   model_options, trng=trng, k=10,
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
                print 'valid set decode done'
                ud_epoch = time.time() - ud_epoch
                ud_epoch /= 60.
                print 'epoch cost time ... ', ud_epoch



            # validate model on validation set and early stop if necessary
            if numpy.mod(uidx, validFreq) == 0:
                use_noise.set_value(0.)
                valid_errs = pred_probs(f_log_probs, prepare_data,
                                        model_options, valid)
                valid_err_cost = valid_errs.mean()
                

                # compute wer
                os.system('python compute-wer.py ' + valid_output[0] + ' ' + valid_datasets[1] + ' ' + valid_result[0])
                fpp=open(valid_result[0])
                stuff=fpp.readlines()
                fpp.close()
                m=re.search('WER (.*)\n',stuff[0])
                valid_per=100. * float(m.group(1))
                m=re.search('ExpRate (.*)\n',stuff[1])
                valid_sacc=100. * float(m.group(1))
                valid_err=valid_per
                #valid_err=0.7*valid_per-0.3*valid_sacc

                history_errs.append(valid_err)

                if uidx/validFreq == 0 or valid_err <= numpy.array(history_errs).min(): # the first time valid or worse model
                    best_p = unzip(tparams)
                    best_bn_p = unzip(bn_tparams)
                    bad_counter = 0

                if uidx/validFreq != 0 and valid_err > numpy.array(history_errs).min():
                    bad_counter += 1
                    if bad_counter > patience:
                        if halfLrFlag==2:
                            print 'Early Stop!'
                            estop = True
                            break
                        else:
                            print 'Lr decay and retrain!'
                            bad_counter = 0
                            lrate = lrate / 2
                            params = best_p
                            bn_params = best_bn_p
                            halfLrFlag += 1 

                if numpy.isnan(valid_err):
                    #ipdb.set_trace()
                    print 'valid_err nan'

                print 'Valid WER: %.2f%%, ExpRate: %.2f%%, Cost: %f' % (valid_per,valid_sacc,valid_err_cost)

            # finish after this many updates
            if uidx >= finish_after:
                print 'Finishing after %d iterations!' % uidx
                estop = True
                break

        print 'Seen %d samples' % n_samples

        if estop:
            break

    if best_p is not None:
        zipp(best_p, tparams)
        zipp(best_bn_p, bn_tparams)

    use_noise.set_value(0.)
    valid_err = pred_probs(f_log_probs, prepare_data,
                           model_options, valid).mean()

    print 'Valid ', valid_err

    params = copy.copy(best_p)
    bn_params = copy.copy(best_bn_p)
    numpy.savez(saveto, zipped_params=best_p,
                history_errs=history_errs,
                **params)
    numpy.savez(bn_saveto, zipped_params=best_bn_p,
                history_errs=history_errs,
                **bn_params)

    return valid_err


if __name__ == '__main__':
    pass
