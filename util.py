# -*- coding: utf-8 -*-
# @Date    : 2019-02-19 16:17:09
# @Author  : Your Name (you@example.org)
# @Link    : http://example.org
# @Version : $1.0$

import os
import math
import numpy as np
from sklearn.metrics import pairwise_kernels
from sklearn.metrics import pairwise_distances

SQRT_CONST = 1e-10


def as_string(desc):
    return ' and '.join(desc)

def refine(seed, candidate, st):
    if seed == '':
        return candidate
    copy = seed + st + candidate
    # copy.append(candidate)
    return copy

def str2list(seed):
    if seed == '':
        return []
    desclist = seed.replace(' and ',';')
    desclist = desclist.replace(' or ',';')
    desclist = desclist.split(';')
    return desclist

def mmd2_rbf(X,Y,sig=1.0):
    """ Computes the l2-RBF MMD for X Y """

    Kxx = np.exp(-pdist2sq(X,X)/np.square(sig))
    Kxy = np.exp(-pdist2sq(X,Y)/np.square(sig))
    Kyy = np.exp(-pdist2sq(Y,Y)/np.square(sig))

    m = np.float(X.shape[0])
    n = np.float(Y.shape[0])

    mmd = 1.0 /(m*(m-1.0))*(Kxx.sum()-m)
    mmd = mmd + 1.0 /(n*(n-1.0))*(Kyy.sum()-n)
    mmd = mmd - 2.0 /(m*n)*Kxy.sum()
#     mmd = 4.0*mmd

    return mmd

def pdist2sq(X,Y):
    """ Computes the squared Euclidean distance between all pairs x in X, y in Y """
    C = -2*np.matmul(X,Y.T)
    nx = np.sum(np.square(X),axis=1,keepdims=True)
    ny = np.sum(np.square(Y),axis=1,keepdims=True)
    D = (C + ny.T) + nx
    return D

def pdist2(X,Y):
    """ Returns the tensorflow pairwise distance matrix """
    return safe_sqrt(pdist2sq(X,Y))

def safe_sqrt(x, lbound=SQRT_CONST):
    ''' Numerically safe version of TensorFlow sqrt '''
    return np.sqrt(np.clip(x, lbound, np.inf))

def MMD2u(K, m, n):
    """The MMD^2_u unbiased statistic.
    """
    Kx = K[:m, :m]
    Ky = K[m:, m:]
    Kxy = K[:m, m:]
    return 1.0 / (m * (m - 1.0)) * (Kx.sum() - Kx.diagonal().sum()) + \
        1.0 / (n * (n - 1.0)) * (Ky.sum() - Ky.diagonal().sum()) - \
        2.0 / (m * n) * Kxy.sum()


def compute_null_distribution(K, m, n, iterations=10000, verbose=False,
                              random_state=None, marker_interval=1000):
    """Compute the bootstrap null-distribution of MMD2u.
    """
    if type(random_state) == type(np.random.RandomState()):
        rng = random_state
    else:
        rng = np.random.RandomState(random_state)

    mmd2u_null = np.zeros(iterations)
    for i in range(iterations):
        if verbose and (i % marker_interval) == 0:
            print(i),
            stdout.flush()
        idx = rng.permutation(m+n)
        K_i = K[idx, idx[:, None]]
        mmd2u_null[i] = MMD2u(K_i, m, n)

    if verbose:
        print("")

    return mmd2u_null

def K_gen(X,Y,kernel_function='rbf'):
	X = X / safe_sqrt(np.sum(np.square(X), axis=1,keepdims=True))
	Y = Y / safe_sqrt(np.sum(np.square(Y), axis=1,keepdims=True))
	m = len(X)
	n = len(Y)
	XY = np.vstack([X, Y])
	K = pairwise_kernels(XY, metric=kernel_function)
	return K,m,n

def MMD_measure(X,Y,h0test=False):
    K,m,n = K_gen(X,Y)
    T_mmd2 = MMD2u(K, m, n)
    if h0test == True:
        iterations=10000
        mmd2u_null = compute_null_distribution(K,m,n,iterations)
        p = max(1.0/iterations, (mmd2u_null > T_mmd2).sum() /
                  float(iterations))
        print('p mmd2u: %f, t mmd2u: %f' % (p,T_mmd2))
    return T_mmd2

### measures

def varphi_ent(d,n):
    d += 0.000001
    n += 0.000002
    return np.sqrt(-(d/n)*np.log2(d/n) - ((n-d)/n)*np.log2((n-d)/n))

def get_infopurity(IDdic, IDdic_sd, IDs):
    info = []
    for id in IDs:
        d = IDdic_sd[id]
        n = IDdic[id]
        info.append(varphi_ent(d, n))
    return np.mean(np.array(info))

def get_sub(sps, ps, N, n):
    sub = []
    # gs = IDgroup[['sp','p']]
    for index,allp in enumerate(ps):
        tp = sps[index]
        fp = n - tp
        alln = N - allp
        tpr = tp / allp
        fpr = fp / alln
        # sub_inf = tpr - fpr
        sub_qlap = (tp + 1) / (tp + fp + 2)
        sub.append(sub_qlap)
    return max(sub)

def get_cvar(IDgroup):
    cvar = 0
    vars = []
    for group_key, group_value in IDgroup:
        group = IDgroup.get_group(group_key)
        labels = group['Y']
        varpred = np.var(labels.to_numpy())
        # varpred = 0
        # for pred in preds:
        #     varpred += (pred - 1)*(pred - 1)
        # varpred = varpred / len(preds)
        vars.append(varpred)
    cvar = np.mean(np.array(vars))
    return cvar
    
def information(sd, fc, P, N):
    dic = sd[fc].value_counts().to_dict()
    if P in dic:
        p = dic[P]
    else: p = 0
    if N in dic:
        n = dic[N]
    else: n = 0
    return p - n

def pred_prob(preds):
    preds = np.array(preds)
    preds= np.sum(np.transpose(preds),axis=1)
    sigmoid = lambda x: 1 / (1 + math.exp(-x))
    results = list(map(sigmoid, preds))
    return results

def pred_cal(rule, data, alpha):
    pred = []
    N = len(data)
    ind = data.eval(rule)
    for i in range(N):
        if ind[i] == False:
            pred.append(-1*alpha)
        else:
            pred.append(alpha)
    return pred

def alpha_cal(rule, data, weights):
    N = len(data)
    ind = data.eval(rule)
    ind = 1*ind
    preds_m = ind.to_numpy()
    err_m = 0
    labels = data['Y'].astype(int).to_numpy()
    for i in range(N):
        if preds_m[i] != labels[i]:
            err_m += weights[i]
    err_m = err_m / np.sum(weights)
    alpha_m = np.log((1-err_m)/(err_m+0.00001))
    return alpha_m

def cvar_cal(data, pred_probs):
    df = data.assign(probs=pred_probs)
    # print(df.head())
    agg_func_math = {
        'probs':
        ['mean', 'var']
    }
    df = df.groupby(['ID','Y']).agg(agg_func_math).round(2)
    # print(df['probs']['var'].mean())
    return df['probs']['var'].mean()