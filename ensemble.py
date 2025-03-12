# -*- coding: utf-8 -*-
# @Date    : 2021-07-21 18:12:14
# @Author  : Your Name (you@example.org)
# @Link    : http://example.org
# @Version : $0.1$

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
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from load import *
from util import *

def boosting_rules(data, data_test, feature, cfg, sdgen, W_prime, logfile, outdir):
    M = cfg['M']
    N = len(data)
    weights = [1/N for i in range(N)]
    dataframe = []
    rounds = []
    alphas = []
    predictors = []
    preds = []
    preds_probs = []
    cvars = []
    cvars_test = []
    rule_tprs = []
    rule_fprs = []
    acc_train = []
    acc_test = []
    tpr_train = []
    tpr_test = []
    fpr_train = []
    fpr_test = []
    acc_maj_train = []
    acc_min_train = []
    acc_maj_test = []
    acc_min_test = []

    labels = data['Y'].astype(int).to_numpy()
    for i in range(M):
        log(logfile, "step : %d" % i)
        rounds.append(i)
        # in_feature = np.random.choice(feature, size=12, replace=False)
        # log(logfile, "features: %s" % in_feature)
        rules = []
        qualities = []
        temptpr = []
        tempcvar = []
        in_data = data.assign(weights=weights)
        # log(logfile, "weights: %s" % in_data.weights.unique())
        EMM_res = sdgen.EMM(in_data, feature, weights, \
            cfg['width'], cfg['depth'], cfg['topq'], cfg['N'], cfg['P'], cfg['split'], cfg['lambda'], \
                            sdgen.eta, sdgen.satisfies_all, sdgen.eval_quality, W_prime, preds, logfile)
        for (desc, quality, coverage) in EMM_res.get_values():
            rules.append(desc)
            # log(logfile, "desc: {}; quality: {}; coverage: {}; bin_quality: {}; sub_quality: {}; tpr: {}; fpr: {}\n".format(
                # desc, quality, coverage[0], coverage[1], coverage[2], coverage[3], coverage[4]))\
            qualities.append(quality)
            temptpr.append((coverage[3], coverage[4]))
            tempcvar.append(coverage[2])
        log(logfile, "quality: %f" % qualities[0])
        rule_tprs.append(temptpr[0][0])
        rule_fprs.append(temptpr[0][1])
        cvars.append(tempcvar[0])
        predictor_m = rules[0]
        ind = data.eval(predictor_m)
        ind = 1*ind
        preds_m = ind.to_numpy()
        err_m = 0
        for i in range(N):
            if preds_m[i] != labels[i]:
                err_m += weights[i]
        err_m = err_m / np.sum(weights)
        alpha_m = np.log((1-err_m)/(err_m + 0.00001))
        # alpha_m = alpha_cal(predictor_m, data, weights)
        alphas.append(alpha_m)
        predictors.append(predictor_m)
        preds.append(pred_cal(predictor_m, data, alpha_m))
        temp = []
        for i in range(N):
            if preds_m[i] != labels[i]:
                temp.append(weights[i] * np.exp(alpha_m))
            else:
                temp.append(weights[i])
        weights = temp
        # for i,pred in enumerate(predictors):
        #     log(logfile, "predictor: %s, alpha: %s" % (pred, alphas[i]))
        train_acc, train_tpr, train_fpr, ac_maj11, ac_maj00, ac_min10, ac_min01 = boosting_eval(data, preds, logfile)
        acc_train.append(train_acc)
        tpr_train.append(train_tpr)
        fpr_train.append(train_fpr)
        acc_maj_train.append((ac_maj11 + ac_maj00)/2)
        acc_min_train.append((ac_min01 + ac_min10)/2)
        log(logfile, "train_acc: %f, train_tpr: %f, train_fpr: %f, train major group1 acc: %f, train major group2 acc: %f, \
            train minor group1 acc: %f, train minor group2 acc: %f, variance: %f" % (train_acc, train_tpr, train_fpr, ac_maj11, ac_maj00, ac_min10, ac_min01, tempcvar[0]))
        test_acc, test_tpr, test_fpr, ac_maj11, ac_maj00, ac_min10, ac_min01, cvartest = boosting_eval_test(data_test, predictors, alphas, logfile)
        acc_test.append(test_acc)
        tpr_test.append(test_tpr)
        fpr_test.append(test_fpr)
        acc_maj_test.append((ac_maj11 + ac_maj00)/2)
        acc_min_test.append((ac_min01 + ac_min10)/2)
        cvars_test.append(cvartest)
        log(logfile, "test_acc: %f, test_tpr: %f, test_fpr: %f, test major group1 acc: %f, test major group2 acc: %f, \
            test minor group1 acc: %f, test minor group2 acc: %f, variance: %f" % (test_acc, test_tpr, test_fpr, ac_maj11, ac_maj00, ac_min10, ac_min01, cvartest))
        # log(logfile, "test_acc: %f, test_tpr: %f, test_fpr: %f" % (test_acc, test_tpr, test_fpr))
        log(logfile, "rule: %s, tpr: %f, fpr: %f" % (predictor_m, temptpr[0][0], temptpr[0][1]))
    dataframe.append(rounds)
    dataframe.append(alphas)
    dataframe.append(predictors)
    dataframe.append(rule_tprs)
    dataframe.append(rule_fprs)
    dataframe.append(acc_train)
    dataframe.append(acc_test)
    dataframe.append(tpr_train)
    dataframe.append(tpr_test)
    dataframe.append(fpr_train)
    dataframe.append(fpr_test)
    dataframe.append(acc_maj_train)
    dataframe.append(acc_maj_test)
    dataframe.append(acc_min_train)
    dataframe.append(acc_min_test)
    # dataframe.append(preds_probs)
    dataframe.append(cvars)
    dataframe.append(cvars_test)
    print(len(preds_probs))
    print(len(cvars))
    print(np.array(dataframe).shape)
    df_out = pd.DataFrame(np.array(dataframe).T, columns =['rounds', 'alphas', 'rules', 'rule tpr', 'rule fpr', 'acc train', 'acc test', 'tpr train', 'tpr test', \
        'fpr train', 'fpr test', 'acc major train', 'acc major test', 'acc minor train', 'acc minor test', 'var train', 'var test'])
    df_out.to_csv(outdir + "statistics_res.csv", sep=',', encoding='utf-8')
    return alphas, predictors, preds

def predict_bag(data, rules, coefs):
    K = len(rules)
    ruleiter = iter(rules)
    rule = next(ruleiter)
    # d_str = as_string(rule)
    ind = data.eval(rule)
    ind = 1*ind
    preds = ind.to_numpy()
    preds = preds.reshape(len(preds),1)
    for rule in ruleiter:
        # d_str = as_string(rule)
        ind = data.eval(rule)
        ind = 1*ind
        ind = ind.to_numpy()
        ind = ind.reshape(len(ind),1)
        preds = np.concatenate([preds, ind], axis=1)
    s = np.array(coefs)
    preds = np.multiply(preds, s)
    results = np.sum(preds, axis=1)
    func = lambda x:0 if x < K/2 else 1
    results = list(map(func, results))
    return results

def eval_acc(data, preds, logfile):
    N = len(preds)
    labels = data['Y'].astype(int).to_numpy()
    acc = accuracy_score(labels, preds)
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    log(logfile, "tn: %f, fp: %f, fn: %f, tp: %f" % (tn, fp, fn, tp))
    tpr = tp / (tp + fn)
    fpr = fp / (tn + fp)
    return acc, tpr, fpr

def boosting_eval(data, preds, logfile):
    labels = data['Y'].astype(int).to_numpy()
    groupID = data['A'].astype(int).to_numpy()
    N = len(data)
    # M = len(predictors)
    # preds = []
    # for j in range(M):
    #     preds_j = []
    #     ind = data.eval(predictors[j])
    #     for i in range(N):
    #         if ind[i] == False:
    #             preds_j.append(-1*alphas[j])
    #         else:
    #             preds_j.append(alphas[j])
    #     preds.append(preds_j)
    # preds = np.array(preds)
    # preds= np.sum(np.transpose(preds),axis=1)
    # results = []
    # for i in range(N):
    #     if preds[i] > 0:
    #         results.append(1)
    #     else:
    #         results.append(0)
    results = np.random.binomial(1, p=pred_prob(preds))
    # print(results)
    # print(labels)
    acc = accuracy_score(labels, results)
    ac_maj11, ac_maj00, ac_min10, ac_min01 = group_err_mnist(labels, results, groupID)
    tn, fp, fn, tp = confusion_matrix(labels, results).ravel()
    # log(logfile, "tn: %f, fp: %f, fn: %f, tp: %f" % (tn, fp, fn, tp))
    tpr = tp / (tp + fn)
    fpr = fp / (tn + fp)
    return acc, tpr, fpr, ac_maj11, ac_maj00, ac_min10, ac_min01

def boosting_eval_test(data, predictors, alphas, logfile):
    labels = data['Y'].astype(int).to_numpy()
    groupID = data['A'].astype(int).to_numpy()
    N = len(data)
    M = len(predictors)
    preds = []
    for j in range(M):
        preds_j = []
        ind = data.eval(predictors[j])
        for i in range(N):
            if ind[i] == False:
                preds_j.append(-1*alphas[j])
            else:
                preds_j.append(alphas[j])
        preds.append(preds_j)
    # preds = np.array(preds)
    # preds= np.sum(np.transpose(preds),axis=1)
    results = np.random.binomial(1, p=pred_prob(preds))
    cvar = cvar_cal(data, results)
    # print(results)
    # print(labels)
    acc = accuracy_score(labels, results)
    ac_maj11, ac_maj00, ac_min10, ac_min01 = group_err_mnist(labels, results, groupID)
    tn, fp, fn, tp = confusion_matrix(labels, results).ravel()
    # log(logfile, "tn: %f, fp: %f, fn: %f, tp: %f" % (tn, fp, fn, tp))
    tpr = tp / (tp + fn)
    fpr = fp / (tn + fp)
    return acc, tpr, fpr, ac_maj11, ac_maj00, ac_min10, ac_min01, cvar

def group_err_mnist(labels, results, groupID):
    maj11 = 1; e_maj11 = 0; maj00 = 1; e_maj00 = 0
    min01 = 1; e_min01 = 0; min10 = 1; e_min10 = 0 

    for i, label in enumerate(labels):
        if groupID[i] == 0:
            maj11 += 1
            maj00 += 1
            if label != results[i]:
                e_maj11 += 1
                e_maj00 += 1
        elif groupID[i] == 1:
            min10 += 1
            min01 +=1
            if label != results[i]:
                e_min10 += 1
                e_min01 += 1
    return (maj11 - e_maj11) / maj11, (maj00 - e_maj00) / maj00, (min10 - e_min10) / min10, (min01 - e_min01) / min01

def group_err(labels, results, groupID):
    maj11 = 0; e_maj11 = 0; maj00 = 0; e_maj00 = 0
    min01 = 0; e_min01 = 0; min10 = 0; e_min10 = 0 

    for i, label in enumerate(labels):
        if groupID[i] == 1 and label == 1:
            maj11 += 1
            if label != results[i]:
                e_maj11 += 1
        elif groupID[i] == -1 and label == 0:
            maj00 += 1
            if label != results[i]:
                e_maj00 += 1
        elif groupID[i] == 1 and label == 0:
            min10 += 1
            if label != results[i]:
                e_min10 += 1
        elif groupID[i] == -1 and label == 1:
            min01 +=1
            if label != results[i]:
                e_min01 += 1
    return (maj11 - e_maj11) / maj11, (maj00 - e_maj00) / maj00, (min10 - e_min10) / min10, (min01 - e_min01) / min01


