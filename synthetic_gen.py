#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-02-03 17:01:46
# @Author  : Your Name (you@example.org)
# @Link    : http://example.org
# @Version : $Id$

import os
import datetime
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append(os.getcwd())

from load import *


def DAG_gen(xn,yn,zn,alpha):
	xs = [str(x) for x in range(xn)]
	ys = [str(y) for y in range(yn)]
	zs = [str(z) for z in range(zn)]
	G = nx.DiGraph()

	for i in xs:
		for j in ys:
			G.add_edge('x' + i, 'y' + j, conf=0)

	for i in ys:
		pn = 0.1
		np.random.shuffle(zs)
		for j in zs:
			p = np.exp(-pn)
			if np.random.binomial(1, p):
				G.add_edge('z'+j, 'y'+i, conf=0)
				pn+=1
			if np.random.binomial(1, alpha):
				G.add_edge('y'+i, 'z'+j, conf=1)

	for i in xs:
		pn = 0.1
		np.random.shuffle(zs)
		for j in zs:
			p = np.exp(-pn)
			if np.random.binomial(1, p):
				G.add_edge('z'+j, 'x'+i, conf=0)
				pn+=1
			if np.random.binomial(1, alpha):
				G.add_edge('x'+i, 'z'+j, conf=1)

	subedges = [edge for edge in G.edges() if G[edge[0]][edge[1]]['conf']==1]

	G_V = G.copy()
	G_V.remove_edges_from(subedges)

	for i in zs:
		pn = 1.5
		if not G_V.has_node('z'+i):
			G_V.add_node('z'+i)
		de_i = nx.descendants(G_V, 'z'+i)
		for j in zs:
			if i==j:continue
			p = np.exp(-pn)
			if 'z'+j not in de_i and np.random.binomial(1, p):
				G.add_edge('z'+j, 'z'+i, conf=0)
				pn+=1
			if np.random.binomial(1, alpha):
				G.add_edge('z'+i, 'z'+j, conf=1)

	return G

	

def main(cfgfile):
	configs = load_config(cfgfile)
	outdir = configs['outdir'][0]
	timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S-%f")
	# os.mkdir(outdir)
	out_DAG = '%s/DAG_%s.csv' % (outdir,timestamp)
	out_df = '%s/data_%s.csv' % (outdir,timestamp)
	cfg = sample_config(configs)
	logfile = outdir+'log.txt'
	f = open(logfile,'w')
	
	log(logfile, ("configs: %s" % str(cfg)))

	dag = DAG_gen(cfg['nx'],cfg['ny'],cfg['nz'],cfg['alpha'])

	f.close()

	# plt.subplot(111)

	# pos=nx.spring_layout(dag)
	# labels=nx.draw_networkx_labels(dag,pos)
	# nodes=nx.draw_networkx_nodes(dag,pos)

	# subedges = [edge for edge in dag.edges() if dag[edge[0]][edge[1]]['conf']==1]
	# nx.draw_networkx_edges(dag, pos, edgelist=subedges, style='dashed', edge_color='b')

	# subedges2 = [edge for edge in dag.edges() if dag[edge[0]][edge[1]]['conf']==0]
	# nx.draw_networkx_edges(dag, pos, edgelist=subedges2, style='solid')

	# plt.show()

	lst = [(edge[0], edge[1], edge[2]['conf']) for edge in dag.edges(data=True)]

	df = pd.DataFrame(lst, columns =['o', 'd', 'conf'])
	df.to_csv(out_DAG, sep=';', encoding='utf-8')



if __name__ == '__main__':
	main(sys.argv[1])