#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2020-02-03 13:48:12
# @Author  : Your Name (you@example.org)
# @Link    : http://example.org
# @Version : $Id$

import os
import sys
sys.path.append(os.getcwd())
import pandas as pd
import numpy as np
import networkx as nx
import sklearn.metrics as metrics

from sklearn.linear_model import LogisticRegression


def DAG_reconstruct(df):
	edgelist = [(row[0],row[1],{'conf':row[2]}) for row in df.values.tolist()]
	G = nx.DiGraph()
	G.add_edges_from(edgelist)
	# if not nx.is_directed_acyclic_graph(G):
	# 	print("Input is not DAG")
	# 	return
	return G

def C_component_decompose(G_An_YX):
	subedges = [edge for edge in G_An_YX.edges() if G_An_YX[edge[0]][edge[1]]['conf']==0]
	G_V_ = G_An_YX.copy()
	G_V_.remove_edges_from(subedges)
	print("nodes: ",G_V_.nodes())
	c_components = list(nx.connected_components(G_V_.to_undirected()))
	print("ccomp:",c_components)
	return c_components

def ancestor_xy(G):
	source = [node for node in G.nodes() if node[0]!='z']
	subedges = [edge for edge in G.edges() if G[edge[0]][edge[1]]['conf']==1]
	G_V = G.copy()
	G_V.remove_edges_from(subedges)
	# print(G_V.edges())
	An_YX_ = []
	for s in source:
		An_YX_.extend(nx.ancestors(G_V, s))
	An_YX_.extend(source)
	An_YX_ = set(An_YX_)
	print("an yx:",An_YX_)
	return An_YX_

def ancestor_y_x_(G):
	source = [node for node in G.nodes() if node[0]=='y']
	subedges = [edge for edge in G.edges() if G[edge[0]][edge[1]]['conf']==1 or edge[0][0]=='x']
	G_V = G.copy()
	G_V.remove_edges_from(subedges)
	# print(G_V.edges())
	An_Y_X_ = []
	for s in source:
		An_Y_X_.extend(nx.ancestors(G_V, s))
	An_Y_X_.extend(source)
	An_Y_X_ = set(An_Y_X_)
	print("An Y:",An_Y_X_)
	return An_Y_X_

def FCS(G):
	W_prime = []
	An_YX = ancestor_xy(G)
	G_An_YX = G.subgraph(An_YX)
	An_Y_X_ = ancestor_y_x_(G_An_YX)
	W = C_component_decompose(G_An_YX)
	for cw in W:
		if len(set(cw).intersection(An_Y_X_))>0:
			W_prime.extend(cw)
	for nn in W_prime:
		if nn[0]!='z':
			W_prime.remove(nn)
	W_prime = set(W_prime)
	print("w`:", W_prime)
	return W_prime