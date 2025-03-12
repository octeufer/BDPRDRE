# -*- coding: utf-8 -*-
# @Date    : 2018-04-02 02:06:14
# @Author  : Your Name (you@example.org)
# @Link    : http://example.org
# @Version : $3.0$

import os
import sys
import datetime
sys.path.append(os.getcwd())
import subgroup_gen as sdgen

from load import *
from dgraph import *
from ensemble import *
from sklearn import metrics

def main(cfgfile):
	configs = load_config(cfgfile)
	outdir = configs['outdir'][0]
	timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S-%f")
	outdir = outdir+'results_'+timestamp+'/'
	os.mkdir(outdir)
	out_file = '%s/results.txt' % outdir
	# output_file = '%s/output.all' % outdir
	cfg = sample_config(configs)
	logfile = outdir+'log.txt'
	f = open(logfile,'w')
	
	log(logfile, ("configs: %s" % str(cfg)))

	dataform = cfg['datadir'] + cfg['dataform']
	dagform = cfg['datadir'] + cfg['DAGform']
	df, feature, edgelist = load_data(dataform, cfg['feature'], dagform, logfile)
	# feature, edgelist = load_data(dataform, cfg['feature'], dagform, logfile)
	datatest = cfg['datadir'] + cfg['datatest']
	df_test, feature, edgelist = load_data(datatest, cfg['feature'], dagform, logfile)

	log(logfile, "features: %s" % feature)

	log(logfile, "data loaded")

	G = DAG_reconstruct(edgelist[["o","d","conf"]])
	W_prime = FCS(G)

	alphas, predictors, preds = boosting_rules(df, df_test, feature, cfg, sdgen, W_prime, logfile, outdir)
	# test_acc, test_tpr, test_fpr = boosting_eval(df_test, predictors, alphas, logfile)
	# log(logfile, "test_accuracy: %f, test_tpr: %f, test_fpr: %f" % (test_acc, test_tpr, test_fpr))
	# acc, tpr, fpr =  boosting_eval(df, predictors, alphas, logfile)
	# log(logfile, "train accuracy: %f, tpr: %f, fpr: %f" % (acc, tpr, fpr))
	
	train_acc, train_tpr, train_fpr, ac_maj11, ac_maj00, ac_min10, ac_min01 = boosting_eval(df, preds, logfile)
	log(logfile, "train_acc: %f, train_tpr: %f, train_fpr: %f, train major group1 acc: %f, train major group2 acc: %f, \
		train minor group1 acc: %f, train minor group2 acc: %f" % (train_acc, train_tpr, train_fpr, ac_maj11, ac_maj00, ac_min10, ac_min01))
	test_acc, test_tpr, test_fpr, ac_maj11, ac_maj00, ac_min10, ac_min01, cvar = boosting_eval_test(df_test, predictors, alphas, logfile)
	log(logfile, "test_acc: %f, test_tpr: %f, test_fpr: %f, test major group1 acc: %f, test major group2 acc: %f, \
		test minor group1 acc: %f, test minor group2 acc: %f" % (test_acc, test_tpr, test_fpr, ac_maj11, ac_maj00, ac_min10, ac_min01))


	# EMM_res = sdgen.EMM(df, feature, \
    #                  cfg['width'], cfg['depth'], cfg['topq'], cfg['N'], cfg['P'], cfg['split'], cfg['lambda'], \
	# 					 sdgen.eta, sdgen.satisfies_all, sdgen.eval_quality, W_prime, logfile)

	# file = open(out_file,"w")
	# # out_desc = ['omega']
	# # out_quality = [-1]
	# # out_coverage = [1.0]
	# # out_embs = [embs_omega]
	# rules = []
	# # tpr, fpr = [], []
	# for (desc, quality, coverage) in EMM_res.get_values():
	# 	rules.append(desc)
	# 	# tpr.append(coverage[3])
	# 	# fpr.append(coverage[4])
	# 	file.write("desc: {}; quality: {}; coverage: {}; bin_quality: {}; sub_quality: {}; tpr: {}; fpr: {}\n".format(
	# 		desc, quality, coverage[0], coverage[1], coverage[2], coverage[3], coverage[4]))
	# 	# out_desc.append(desc)
	# 	# out_quality.append(quality)
	# 	# out_coverage.append(coverage)
	# 	# out_embs.append(embs_sd)
	# M = len(rules)
	# coefs = [1 for i in range(M)]
	# train_preds = predict(df, rules, coefs)
	# acc, tpr, fpr = eval_acc(df, train_preds, logfile)
	# log(logfile, "train accuracy: %f, tpr: %f, fpr: %f" % (acc, tpr, fpr))

	# test_preds = predict(df_test, rules, coefs)
	# test_acc, test_tpr, test_fpr = eval_acc(df_test, test_preds, logfile)
	# log(logfile, "test_accuracy: %f, test_tpr: %f, test_fpr: %f" % (test_acc, test_tpr, test_fpr))

	# file.close()
	# pikcle.dump(output, output_file)
	# np.savez(output_file, desc = np.array(out_desc), quality = np.array(out_quality), \
	# 	coverage = np.array(out_coverage), embs = np.array(out_embs))
	f.close()

if __name__ == '__main__':
	main(sys.argv[1])
