import theano
import theano.tensor as tensor

import numpy

profile = False

def itemlist(tparams):
    return [vv for kk, vv in tparams.iteritems()]

def itemlist_name(tparams):
    return [kk for kk, vv in tparams.iteritems()]

"""
General Optimizer Structure: (adadelta, adam, rmsprop, sgd)
Parameters
----------
    lr : theano shared variable
        learning rate, currently only necessaary for sgd
    tparams : OrderedDict()
        dictionary of shared variables {name: variable}
    grads : 
        dictionary of gradients
    inputs :
        inputs required to compute gradients
    cost : 
        objective of optimization
    hard_attn_up :
        additional updates required for hard attention mechanism learning 
Returns
-------
    f_grad_shared : compute cost, update optimizer shared variables
    f_update : update parameters
"""
# optimizers
# name(hyperp, tparams, grads, inputs (list), cost) = f_grad_shared, f_update
def adam(lr, tparams, bn_tparams, opt_ret, grads, inp, cost):
    gshared = [theano.shared(p.get_value() * 0.,
                             name='%s_grad' % k)
               for k, p in tparams.iteritems()]
    rule = []
    for idx in bn_tparams:
        rule += [(bn_tparams[idx], opt_ret[idx])]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]
    f_grad_shared = theano.function(inp, cost, updates=rule+gsup, profile=profile)

    #lr0 = 0.0002
    lr0 = lr
    b1 = 0.1
    b2 = 0.001
    e = 1e-8

    updates = []

    i = theano.shared(numpy.float32(0.))
    i_t = i + 1.
    fix1 = 1. - b1**(i_t)
    fix2 = 1. - b2**(i_t)
    lr_t = lr0 * (tensor.sqrt(fix2) / fix1)

    for p, g in zip(tparams.values(), gshared):
        m = theano.shared(p.get_value() * 0.)
        v = theano.shared(p.get_value() * 0.)
        m_t = (b1 * g) + ((1. - b1) * m)
        v_t = (b2 * tensor.sqr(g)) + ((1. - b2) * v)
        g_t = m_t / (tensor.sqrt(v_t) + e)
        p_t = p - (lr_t * g_t)
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
    updates.append((i, i_t))

    f_update = theano.function([lr], [], updates=updates,
                               on_unused_input='ignore', profile=profile)

    return f_grad_shared, f_update

def adam_weightnoise(lr, tparams_miu, tparams_sigma, bn_tparams, opt_ret, grads_miu, grads_sigma, inp, cost):
    gshared_miu = [theano.shared(p.get_value() * 0.,
                             name='%s_grad' % k)
               for k, p in tparams_miu.iteritems()]
    gshared_sigma = [theano.shared(p.get_value() * 0.,
                             name='%s_grad' % k)
               for k, p in tparams_sigma.iteritems()]
    rule = []
    for idx in bn_tparams:
        rule += [(bn_tparams[idx], opt_ret[idx])]
    gsup_miu = [(gs, g) for gs, g in zip(gshared_miu, grads_miu)]
    gsup_sigma = [(gs, g) for gs, g in zip(gshared_sigma, grads_sigma)]
    f_grad_shared = theano.function(inp, cost, updates=rule+gsup_miu+gsup_sigma, profile=profile)

    #lr0 = 0.0002
    lr0 = lr
    b1 = 0.1
    b2 = 0.001
    e = 1e-8

    updates_miu = []
    updates_sigma = []

    i_miu = theano.shared(numpy.float32(0.))
    i_t_miu = i_miu + 1.
    fix1_miu = 1. - b1**(i_t_miu)
    fix2_miu = 1. - b2**(i_t_miu)
    lr_t_miu = lr0 * (tensor.sqrt(fix2_miu) / fix1_miu)

    for p_miu, g_miu in zip(tparams_miu.values(), gshared_miu):
        m_miu = theano.shared(p_miu.get_value() * 0.)
        v_miu = theano.shared(p_miu.get_value() * 0.)
        m_t_miu = (b1 * g_miu) + ((1. - b1) * m_miu)
        v_t_miu = (b2 * tensor.sqr(g_miu)) + ((1. - b2) * v_miu)
        g_t_miu = m_t_miu / (tensor.sqrt(v_t_miu) + e)
        p_t_miu = p_miu - (lr_t_miu * g_t_miu)
        updates_miu.append((m_miu, m_t_miu))
        updates_miu.append((v_miu, v_t_miu))
        updates_miu.append((p_miu, p_t_miu))
    updates_miu.append((i_miu, i_t_miu))
    f_update_miu = theano.function([lr], [], updates=updates_miu,
                               on_unused_input='ignore', profile=profile)

    i_sigma = theano.shared(numpy.float32(0.))
    i_t_sigma = i_sigma + 1.
    fix1_sigma = 1. - b1**(i_t_sigma)
    fix2_sigma = 1. - b2**(i_t_sigma)
    lr_t_sigma = lr0 * (tensor.sqrt(fix2_sigma) / fix1_sigma)
    for p_sigma, g_sigma in zip(tparams_sigma.values(), gshared_sigma):
        m_sigma = theano.shared(p_sigma.get_value() * 0.)
        v_sigma = theano.shared(p_sigma.get_value() * 0.)
        m_t_sigma = (b1 * g_sigma) + ((1. - b1) * m_sigma)
        v_t_sigma = (b2 * tensor.sqr(g_sigma)) + ((1. - b2) * v_sigma)
        g_t_sigma = m_t_sigma / (tensor.sqrt(v_t_sigma) + e)
        p_t_sigma = p_sigma - (lr_t_sigma * g_t_sigma)
        updates_sigma.append((m_sigma, m_t_sigma))
        updates_sigma.append((v_sigma, v_t_sigma))
        updates_sigma.append((p_sigma, p_t_sigma))
    updates_sigma.append((i_sigma, i_t_sigma))
    f_update_sigma = theano.function([lr], [], updates=updates_sigma,
                               on_unused_input='ignore', profile=profile)

    return f_grad_shared, f_update_miu, f_update_sigma

def sgd_momentum(lr, tparams, bn_tparams, opt_ret, grads, inp, cost):
    gshared = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_grad' % k)
               for k, p in tparams.iteritems()]
    prev_update = [theano.shared(p.get_value() * numpy.float32(0.), name='%s_update' % k)
               for k, p in tparams.iteritems()]
    rule = []
    for idx in bn_tparams:
        rule += [(bn_tparams[idx], opt_ret[idx])]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]
    f_grad_shared = theano.function(inp, cost, updates=rule+gsup, profile=False)      # calculate the model

    # Save the update_delta[i] (this time) 
    pup1 = [(delta, 0.9*delta - lr * g) for delta, g in zip(prev_update, gshared)] # |0.9*delta| why the result can't converge
    pup2 = [(p, p + delta) for p, delta in zip(itemlist(tparams), prev_update)]
    f_update = theano.function([lr], [], updates=pup1+pup2, profile=False)
    return f_grad_shared, f_update

def adadelta(lr, tparams, bn_tparams, opt_ret, grads, inp, cost):
    zipped_grads = [theano.shared(p.get_value() * numpy.float32(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.iteritems()]
    running_up2 = [theano.shared(p.get_value() * numpy.float32(0.),
                                 name='%s_rup2' % k)
                   for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy.float32(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.iteritems()]

    rule = []
    for idx in bn_tparams:
        rule += [(bn_tparams[idx], opt_ret[idx])]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function(inp, cost, updates=rule+zgup+rg2up,
                                    profile=profile)

    updir = [-tensor.sqrt(ru2 + lr) / tensor.sqrt(rg2 + lr) * zg
             for zg, ru2, rg2 in zip(zipped_grads, running_up2,
                                     running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(itemlist(tparams), updir)]

    f_update = theano.function([lr], [], updates=ru2up+param_up,
                               on_unused_input='ignore', profile=profile)

    return f_grad_shared, f_update

def adadelta_weightnoise(lr, tparams_miu, tparams_sigma, bn_tparams, opt_ret, grads_miu, grads_sigma, inp, cost):
    zipped_grads_miu = [theano.shared(p.get_value() * numpy.float32(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams_miu.iteritems()]
    running_up2_miu = [theano.shared(p.get_value() * numpy.float32(0.),
                                 name='%s_rup2' % k)
                   for k, p in tparams_miu.iteritems()]
    running_grads2_miu = [theano.shared(p.get_value() * numpy.float32(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams_miu.iteritems()]
    zipped_grads_sigma = [theano.shared(p.get_value() * numpy.float32(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams_sigma.iteritems()]
    running_up2_sigma = [theano.shared(p.get_value() * numpy.float32(0.),
                                 name='%s_rup2' % k)
                   for k, p in tparams_sigma.iteritems()]
    running_grads2_sigma = [theano.shared(p.get_value() * numpy.float32(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams_sigma.iteritems()]
    
    rule = []
    for idx in bn_tparams:
        rule += [(bn_tparams[idx], opt_ret[idx])]

    zgup_miu = [(zg, g) for zg, g in zip(zipped_grads_miu, grads_miu)]
    rg2up_miu = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2_miu, grads_miu)]
    zgup_sigma = [(zg, g) for zg, g in zip(zipped_grads_sigma, grads_sigma)]
    rg2up_sigma = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2_sigma, grads_sigma)]
    f_grad_shared = theano.function(inp, cost, updates=rule+zgup_miu+rg2up_miu+zgup_sigma+rg2up_sigma,
                                    profile=profile)

    updir_miu = [-tensor.sqrt(ru2 + lr) / tensor.sqrt(rg2 + lr) * zg
             for zg, ru2, rg2 in zip(zipped_grads_miu, running_up2_miu,
                                     running_grads2_miu)]
    ru2up_miu = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2_miu, updir_miu)]
    param_up_miu = [(p, p + ud) for p, ud in zip(itemlist(tparams_miu), updir_miu)]

    f_update_miu = theano.function([lr], [], updates=ru2up_miu+param_up_miu,
                               on_unused_input='ignore', profile=profile)

    updir_sigma = [-tensor.sqrt(ru2 + lr) / tensor.sqrt(rg2 + lr) * zg
             for zg, ru2, rg2 in zip(zipped_grads_sigma, running_up2_sigma,
                                     running_grads2_sigma)]
    ru2up_sigma = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2_sigma, updir_sigma)]
    param_up_sigma = [(p, p + ud) for p, ud in zip(itemlist(tparams_sigma), updir_sigma)]

    f_update_sigma = theano.function([lr], [], updates=ru2up_sigma+param_up_sigma,
                               on_unused_input='ignore', profile=profile)
    return f_grad_shared, f_update_miu, f_update_sigma
