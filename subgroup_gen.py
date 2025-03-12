# -*- coding: utf-8 -*-
# @Date    : 2018-04-02 02:06:14
# @Author  : Your Name (you@example.org)
# @Link    : http://example.org
# @Version : $3.0$

import os
import sys
sys.path.append(os.getcwd())
import pandas as pd
import numpy as np
import re
import datetime
import math
import time

from numpy import linalg as LA
from load import *
from util import *

# print(df.head())
# embeddings=dict()

# print()
# for f in features:
#     print("{}: {}".format(f, df.loc[0, f]))

# print()
# print(df.columns.to_series().groupby(df.dtypes).groups)

import heapq

key_labels = {}

#
class BoundedPriorityQueue:
    """
    Ensures uniqness
    Keeps a maximum size (throws away value with least quality)
    """

    def __init__(self, bound, df):
        self.values = []
        self.bound = bound
        self.entry_count = 0
        self.df = df

    def desc_intersect(self, desc1, coverage, desc2, c):
        ind_new = self.df.eval(as_string(desc1))
        ind_old = self.df.eval(as_string(desc2))
        # if coverage*0.9 > c:
        #     return False
        if (ind_new & ind_old).sum() > c*0.8:
            return True
        return False

    def add(self, element, quality, coverage):
        if any((e == element for (_, _, e, _) in self.values)):
            return  # avoid duplicates

        if any((qq == quality for (qq, _, _, _) in self.values)):
            return  # avoid duplicates

        # if any((self.desc_intersect(element, coverage, e, c) for (_,_,e,c) in self.values)):
        #     return        

        new_entry = (quality, self.entry_count, element, coverage)
        if (len(self.values) >= self.bound):
            temp=heapq.heappushpop(self.values, new_entry)
        else:
            heapq.heappush(self.values, new_entry)

        self.entry_count += 1

    def get_values(self):
        for (q, _, e, coverage) in sorted(self.values, reverse=True):
            yield (e, q, coverage)

    def out_vectors(self):
        return [q for (q, _, _, _) in sorted(self.values, reverse=True)]

    def show_contents(self,logfile):  # for debugging
        log(logfile, "show_contents")
        for (q, entry_count, e, coverage) in self.values:
            # print(q, entry_count, e)
            log(logfile, "quality: %f, entry: %d, element: %s, coverage: %f" % (q, entry_count, e, coverage[0]))

#
class Queue:
    """
    Ensures uniqness
    """

    def __init__(self):
        self.items = []

    def is_empty(self):
        return self.items == []

    def enqueue(self, item):
        if item not in self.items:
            self.items.insert(0, item)

    def dequeue(self):
        return self.items.pop()

    def size(self):
        return len(self.items)

    def get_values(self):
        return self.items

    def add_all(self, iterable):
        for item in iterable:
            self.enqueue(item)

    def clear(self):
        self.items.clear()

#


def EMM(df, fs, weights, w, d, q, NN, PP, sp, lambdaa, eta, satisfies_all, eval_quality, W_prime, preds, logfile):
    """
    w - width of beam
    d - num levels
    q - max results
    eta - a function that receives a description and returns all possible refinements
    satisfies_all - a function that receives a description and verifies wheather it satisfies some requirements as needed
    eval_quality - returns a quality for a given description. This should be comparable to qualities of other descriptions
    catch_all_description - the equivalent of True, or all, as that the whole dataset shall match
    """
    N = df.shape[0]

    ydic = df['Y'].value_counts().to_dict()
    AllP = ydic[PP]
    AllN = ydic[NN]

    # IDdic = df['ID'].value_counts().to_dict()
    # IDs = df['ID'].unique()

    # df_test = df.iloc[sp:]
    # ytdic = df_test['Y'].value_counts().to_dict()
    # test_P = ytdic[PP]
    # test_N = ytdic[NN]

    # for f in fs:
    #     if f not in W_prime:
    #         f.remove(f)
    # IDgroup = df.groupby(['ID', 'Y'])
                
    resultSet = BoundedPriorityQueue(q,df)
    candidateQueue = Queue()
    candidateQueue.enqueue('')
    for level in range(d):
        # log(logfile, "level : %d" % level)
        beam = BoundedPriorityQueue(w,df)
        for seed in candidateQueue.get_values():
            # log(logfile, "seed : %s" % seed)
            for desc in eta(seed,fs,df):
                if satisfies_all(desc,df,N):
                    quality, coverage = eval_quality(
                        desc, df, N, PP, NN, AllP, AllN, lambdaa, logfile, weights, preds)
                    resultSet.add(desc, quality, coverage)
                    beam.add(desc, quality, coverage)
        # beam.show_contents(logfile)
        #candidateQueue.clear()
        candidateQueue = Queue()
        candidateQueue.add_all(desc for (desc, _, _) in beam.get_values())
    return resultSet

####################################################################################################################################

# def refine(desc, more):
#     copy = desc[:]
#     copy.append(more)
#     return copy

def eta(seed,fs,df):
    desclist = str2list(seed)
    for f in fs:
        column_data = df[f]
        if (df[f].dtype == 'float64'):
            min_val, max_val = min(column_data), max(column_data)
            for x in np.linspace(min_val, max_val, 16):
                candidate = "{} <= {}".format(f, x)
                if not candidate in desclist: # if not already there
                    yield refine(seed, candidate, " and ")
                    yield refine(seed, candidate, " or ")
                candidate = "{} > {}".format(f, x)
                if not candidate in desclist: # if not already there
                    yield refine(seed, candidate, " and ")
                    yield refine(seed, candidate, " or ")
        elif (df[f].dtype == 'object'):
            uniq = column_data.dropna().unique()
            for i in uniq:
                candidate = "{} == '{}'".format(f, i)
                if not candidate in desclist: # if not already there
                    yield refine(seed, candidate, " and ")
                    yield refine(seed, candidate, " or ")
                candidate = "{} != '{}'".format(f, i)
                if not candidate in desclist: # if not already there
                    yield refine(seed, candidate, " and ")
                    yield refine(seed, candidate, " or ")
        elif (df[f].dtype == 'int64'):
            min_val, max_val = min(column_data), max(column_data)
            for x in np.linspace(min_val, max_val, 8):
                candidate = "{} <= {}".format(f, np.floor(x))
                if not candidate in desclist: # if not already there
                    yield refine(seed, candidate, " and ")
                    yield refine(seed, candidate, " or ")
                candidate = "{} > {}".format(f, np.floor(x))
                if not candidate in desclist: # if not already there
                    yield refine(seed, candidate, " and ")
                    yield refine(seed, candidate, " or ")
        elif (df[f].dtype == 'bool'):
            uniq = column_data.dropna().unique()
            for i in uniq:
                candidate = "{} == '{}'".format(f, i)
                if not candidate in desclist: # if not already there
                    yield refine(seed, candidate, " and ")
                    yield refine(seed, candidate, " or ")
                candidate = "{} != '{}'".format(f, i)
                if not candidate in desclist: # if not already there
                    yield refine(seed, candidate, " and ")
                    yield refine(seed, candidate, " or ")                    
        else:
            assert False

def satisfies_all(desc,df,N):
    # d_str = as_string(desc)
    ind = df.eval(desc)
    cover_desc = sum(ind)
    return (cover_desc / N > 0.1)
    # return cover_desc

def eval_quality(desc, df, N, PP, NN, AllP, AllN, lambdaa, logfile, weights, preds):
    # d_str = as_string(desc)
    ind = df.eval(desc)
    
    alpha = alpha_cal(desc, df, weights)
    temp = preds.copy()
    temp.append(pred_cal(desc, df, alpha))
    pred_probs = pred_prob(temp)
    cvar = cvar_cal(df, pred_probs)

    df_sd = df.loc[ind]
    nn = df_sd.shape[0]
    coverage = nn / N

    p0 = 0
    n0 = 0
    ydic = df_sd['Y'].value_counts().to_dict()
    if PP in ydic:
        p0 = ydic[PP]
    if NN in ydic:
        n0 = ydic[NN]

    p = 0
    n = 0

    # for index, row in df_sd.iterrows():
    #     if row['Y'] == PP:
    #         p += weights[index]
    #     else:
    #         n += weights[index]

    # ERM
    # dff = df_sd.groupby('Y').sum()
    # if PP in dff.index:
    #     p = dff.loc[PP]['weights']
    # if NN in dff.index:
    #     n = dff.loc[NN]['weights']

    # q_lap = (p + 1) / (p + n + 2)

    # DRO
    dff = df_sd.groupby(['A', 'Y']).sum()
    pp0 = 0
    nn0 = 0
    if (0, 1) in dff.index:
        pp0 = dff.loc[0].loc[PP]['weights']
    if (0, 0) in dff.index:
        nn0 = dff.loc[0].loc[NN]['weights']
    q_lap0 = (pp0 + 1) / (pp0 + nn0 + 2)
    
    pp1 = 0
    nn1 = 0
    if (1, 1) in dff.index:
        pp1 = dff.loc[1].loc[PP]['weights']
    if (1, 0) in dff.index:
        nn1 = dff.loc[1].loc[NN]['weights']
    q_lap1 = (pp1 + 1) / (pp1 + nn1 + 2)
    
    if q_lap0 > q_lap1:
        q_lap = q_lap1
    else:
        q_lap = q_lap0
    # q_lap = p-n

    # cvar = get_cvar(IDgroup)

    # information purity
    # IDdic_sd = df_sd['ID'].value_counts().to_dict()
    # cvar = get_infopurity(IDdic, IDdic_sd, IDs)

    # sp, sn = spu_count_cmnist(df_sd)

    # q_inf = p/AllP - n/AllN
    
    # q_lap_test = (sp + 1) / (sp + sn + 2)

    ###subquality
    # IDgroup = df_sd[['ID','Y']].value_counts().to_dict()
    # groups = df[['ID','Y']].value_counts().to_dict()
    # sps = []
    # ps = []
    # for key in groups.keys():
    #     if key[1] == NN:
    #         continue
    #     if key in IDgroup:
    #         sps.append(IDgroup[key])
    #         ps.append(groups[key])
    #     else:
    #         sps.append(0)
    #         ps.append(groups[key])
    # sub_lap = get_sub(sps, ps, N, nn)

    # if q_lap > 0.7 and cvar < 0.1:
    #     log(logfile, "rule: %s, quality: %f, cvar: %f, coverage: %f" %
    #         (d_str, q_lap, cvar, coverage))

    #temp placeholder

    # ent = varphi_ent(sum(ind),N)
    # return (q_lap - lambdaa * cvar), (coverage, q_lap, cvar, p0/AllP, n0/AllN)
    # return qsd, coverage
    # return q_lap, (coverage, p/nn, n/nn, sp/spunn, sn/spunn)
    return q_lap, (coverage, q_lap, cvar, p0/AllP, n0/AllN)

def spu_count(df_sd):
    sp = 0
    sn = 0
    for index, row in df_sd.iterrows():
        if row['A'] == row['Y']:
            continue
        if row['Y'] == 1:
            sp+=1
        else:
            sn+=1
    return sp, sn


def spu_count_cmnist(df_sd):
    sp = 0
    sn = 0
    for index, row in df_sd.iterrows():
        if index < 50000:
            continue
        if row['Y'] == 1:
            sp += 1
        else:
            sn += 1
    return sp, sn

def spu_count_waterbirds(df_sd):
    sp = 0
    sn = 0
    for index, row in df_sd.iterrows():
        # if row['g'] == 1 or row['g'] == 2:
        if row['split'] == 0:
            continue
        if row['Y'] == 1:
            sp += 1
        else:
            sn += 1
    return sp, sn


