#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 11:12:08 2019

@author: marina
"""

import networkx as nx
from networkx import edge_betweenness_centrality as betweenness
def example_graph():
    """
    Create the example graph from class. Used for testing.
    Do not modify.
    """
    g = nx.Graph()
    #g.add_edges_from([('A', 'B'), ('A', 'D'), ('A', 'T'), ('A', 'Z'), ('B', 'C'), ('D', 'C'), ('T', 'F'), ('Z', 'F'), ('F', 'E'),('C', 'E')])
    g.add_edges_from([('A', 'B'), ('A', 'C'), ('B', 'C'), ('B', 'D'), ('D', 'E'), ('D', 'F'), ('D', 'G'), ('E', 'F'), ('G', 'F')])

    return g
def most_central_edge(G):
    
    centrality = betweenness(G, weight='weight')
    print("Centrality", centrality)