#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-10-06 16:45:17
# @Author  : 
# @Link    : http://example.org
# @Version : $0.1$

import os
import sys
sys.path.append(os.getcwd())
import ast
import subgroup_gen
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

res_file = "reports\\karate5.txt"
# infile = "..\\Code\\data\\football_x4_10k.csv"
infile = "..\\Code\\data\\karate_x4_10k.csv"

def read_result(result_file,df):
    file = open(result_file,'r')
    dqlist = dict()
    # file.readline() #skip the first line
    for line in file.readlines():
        items = line.strip().split(';')
        desc = ast.literal_eval(items[0])
        d_str = subgroup_gen.as_string(desc)
        ind = df.eval(d_str)
        dqlist[items[0]]=(float(items[1]),ind)
    return dqlist

def read_rand(infile):
	df = pd.read_csv(infile, sep=';', encoding='utf-8')
	df['o'] = df['o'].astype('int64')
	df['d'] = df['d'].astype('int64')
	print(df.dtypes)
	return df

def rand_baseline(df):
	rand_nx = df.loc[(df.x1>=5) & (df.x2>=5)]

def confumat(dqlist,df):
	tn, fp, fn, tp = 0,0,0,0
	ind_tref = ((df.x1<5) & (df.x2<5)) | ((df.x1>=5) & (df.x2>=5))
	true_labels = sum(ind_tref)
	print(true_labels)
	false_labels = len(df) - true_labels
	print(false_labels)
	# print(type(ind_tref))
	Q = len(dqlist.keys())
	for value in dqlist.values():
		ind_test = value[1]
		tn1, fp1, fn1, tp1 = confusion_matrix(ind_tref, ind_test).ravel()
		tn += tn1
		fp += fp1
		fn += fn1
		tp += tp1
	avg_tn, avg_fp, avg_fn, avg_tp = tn/Q, fp/Q, fn/Q, tp/Q
	print("tp: %d, tn: %d, fp: %d, fn: %d" % (avg_tp,avg_tn,avg_fp,avg_fn))
	tpr,tnr,fpr,fnr = avg_tp/true_labels, avg_tn/false_labels, avg_fp/false_labels, avg_fn/true_labels
	ppv = avg_tp / (avg_tp + avg_fp)
	return tpr,tnr,fpr,fnr,ppv


def main():
	df = read_rand(infile)
	dqlist = read_result(res_file,df)
	tpr,tnr,fpr,fnr,ppv = confumat(dqlist,df)
	print("tpr: %f,tnr: %f,fpr: %f,fnr: %f,ppv: %f" % (tpr,tnr,fpr,fnr,ppv))

if __name__ == '__main__':
	main()
