#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 10:36:37 2019

@author: marina
"""
from collections import Counter, defaultdict, deque
import copy
from itertools import combinations
import math
import pickle
import networkx as nx
import urllib.request
import operator
from TwitterAPI import TwitterAPI
import time
import sys


#YO NUEVAS
consumer_key = 'yuBdTI32ESnZYdLuJAElEGvL5'
consumer_secret = 'DNycDzVl3Yeqleog1DyiWMYKjBHjmo5S5kgw05yndMMyVfeJ35'
access_token = '1090407968028459008-R9ZYN3WrmFSyOVtN4SGTVFtD75BI58'
access_token_secret = 'OciUhBYBZp67foDHVMZuuirIC2aLGykhrsRjK3dCWngBK'
# =============================================================================
# #JOSEPA
# consumer_key = 'RRkJlRIsNGvsZ9E1GW8V5MRXP'
# consumer_secret = 'jxmWkddzHgVSh5gsCqRPOUCBx5OCFdTAtMzhgqPdUIBNatDr7a'
# access_token = '3020342014-gwUKeKgKGKTxP6xxFrRDec5Pwi6uE2gEZUlHie1'
# access_token_secret = 'Vaqmg0cCa3hpQQT3zbApv2bzsiTZchbtX5xHDl8siqOMb'
# =============================================================================
# =============================================================================
# 
# #MANU
# consumer_key = 'R1Fc15NNPRwQYYNl31N4UQvkW'
# consumer_secret = '7rDpmsIQJjr5B5AlvP1j5IBIG3aWbGrNL6xWBNG3cKQeQmawRd'
# access_token = '1088134586020839424-GFkAElFoEG5dAdH0hH3dFHrnhjbD2K'
# access_token_secret = 'isgVsLChdIQRSZw6raq646mEuMSPayNa3M13UC2Vkijhn'
# =============================================================================

# =============================================================================
# consumer_key = 'QHYcvYjY3HtkLCVksaKGZfdAe'
# consumer_secret = 'Yz3hNqDL9VcLDq24eCMhBHyHyXBYwJDK0EVPFQnA3AS7PFA2I3'
# access_token = '1090407968028459008-NOFwbMEOON0Dn8LYwx36LIdq0xWRr2'
# access_token_secret = 'wHG8hTHajd29zbOpbGlel3tPte0c0wnY34wcWI6Gc0hgs'
# =============================================================================


# =============================================================================
# Number of users collected:
# Number of messages collected:
# Number of communities discovered:
# Average number of users per community:
# Number of instances per class found:
# One example from each class:
# =============================================================================

def load_graph():
    graph = pickle.load(open('./data/graph.txt','rb'))
    return graph
    
def bfs(graph, root, max_depth):

    depth = 0
    q = deque()
    seen = set()  
    node2distances = defaultdict(int)
    node2num_paths = defaultdict(int)
    node2parents = defaultdict(list)

    node2distances[root] = depth
    node2num_paths[root] = 1
    q.append(root)
    
    while len(q) > 0:  
        n = q.popleft()
        depth = node2distances[n]+1
        if depth > max_depth:
            break
        if n not in seen:
            seen.add(n)
            for nn in graph.neighbors(n):
 
                if nn not in seen:
                    q.append(nn)
                    if(not node2distances[nn]):
                        node2distances[nn] = depth
                    if(node2distances[nn]>node2distances[n]):
                        node2parents[nn].append(n)
                        node2num_paths[nn]+=node2num_paths[n] 

    return node2distances, node2num_paths, node2parents
      
    
def complexity_of_bfs(V, E, K):

    r = min(V + E, K)
    return r


def bottom_up(root, node2distances, node2num_paths, node2parents):

    nodes_credit = defaultdict(float)
    edges_credit = defaultdict(float)
    children = defaultdict(list)
  
    for node, parents in node2parents.items():
        for parent in parents:
            children[parent].append(node)
      
    for node,d in node2distances.items():
        for c in children[node]:
            nodes_credit[c] = 1
                
    ordered = dict(sorted(node2distances.items(), key=lambda kv: kv[1], reverse=True))
    for node, d in ordered.items():        
        if children[node]:        
            for c in children[node]:
                n1 = sorted([c, node])[0]
                n2 = sorted([c, node])[1]
                edges_credit[(n1,n2)] = nodes_credit[c]/len(node2parents[c])
                nodes_credit[node] += edges_credit[(n1,n2)]
        
    return edges_credit                                          
            
   
   
    
def approximate_betweenness(graph, max_depth):

    betweenness = defaultdict(float)
    
    
    for node1 in graph.nodes():
        node2distances, node2num_paths, node2parents = bfs(graph, node1, max_depth)
        credit = bottom_up(node1, node2distances, node2num_paths, node2parents)
        for e, b in credit.items():
            betweenness[e] += b/2
            

    return betweenness

def get_components(graph):

    return [c for c in nx.connected_component_subgraphs(graph)]

def partition_girvan_newman(graph, max_depth):

    graph_copy = graph.copy()
    components = []
    betweenness = approximate_betweenness(graph, max_depth)
    betweenness = sorted(betweenness.items(), key = operator.itemgetter(1), reverse = True)
    
    for element in betweenness:
        edge = element[0]
        graph_copy.remove_edge(edge[0], edge[1])
        if(len(get_components(graph_copy))>1):
            components = get_components(graph_copy)
            components = sorted(components, key=lambda x: sorted(x.nodes())[0])
            return components
        else:
            continue
        
    return components

    
def get_subgraph(graph, min_degree):

    subgraph = graph.copy()
    
    for node in graph.nodes():
        degree = graph.degree[node]
        if degree < min_degree:
            subgraph.remove_node(node)
            
    return subgraph


def volume(nodes, graph):
    volume = 0
    for edge in graph.edges():
        for n in nodes:
            if(edge[0] == n or edge[1] == n):
                volume += 1
                break
    
    return volume
                


def cut(S, T, graph):

    
    cut_set = 0
    
    
    for edge in graph.edges(): 
        nodeS = False
        nodeT = False
        for s in S:
            if (s == edge[0] or s == edge[1]):
                nodeS = True
                
        for t in T:
            if (t == edge[0] or t == edge[1]):
                nodeT = True
                
        if nodeS == True and nodeT == True:
            cut_set += 1
      
    
    return cut_set
            
                


def norm_cut(S, T, graph):
    ncv = 0
    
    cutST = cut(S, T, graph)
    volS = volume(S, graph)
    
    volT = volume(T, graph)
    ncv = (cutST/volS) + (cutST/volT)
    return ncv
    


def brute_force_norm_cut(graph, max_size):

    comb = combinations(graph.edges(), max_size)
    elms = [list(x) for x in comb]
    result = []
    clusters = 0
    
    for edge in graph.edges():
        edge_list = []
        graph2 = graph.copy()
        graph2.remove_edge(edge[0], edge[1])
        clusters = len(get_components(graph2))
        if clusters == 2:
            score = norm_cut(get_components(graph2)[0].nodes(), get_components(graph2)[1].nodes(), graph)
            edge_list.append(edge)
            result.append((score, edge_list))
    
    if max_size > 1:
        for elm in elms:
            edge_list = []    
            graph2 = graph.copy()
            for edge in elm:            
                graph2.remove_edge(edge[0],edge[1])
                edge_list.append(edge)
                clusters = len(get_components(graph2))
            if clusters == 2:
                score = norm_cut(get_components(graph2)[0].nodes(), get_components(graph2)[1].nodes(), graph)
                if([score,edge_list] not in result):
                    result.append((score,edge_list))
    return result
    
    
def score_max_depths(graph, max_depths):

    result = []
    max_depth = 0
    
    for md in max_depths:
       max_depth = md
       partitions = partition_girvan_newman(graph, max_depth)
       ncv = norm_cut(partitions[0],partitions[1],graph)
       result.append((max_depth,ncv))
       
    return result



def make_training_graph(graph, test_node, n):

    graphCopy = graph.copy()
    neigh = sorted(graph.neighbors(test_node))[:n]
    for n in neigh:
      graphCopy.remove_edge(test_node, n)
    return graphCopy
        
       
def jaccard(graph, node, k):

    neighbors = set(graph.neighbors(node))
    scores = []
    result = []
    for n in graph.nodes():
        if n!=node:
            neighbors2 = set(graph.neighbors(n))
            score = len(neighbors & neighbors2)/len(neighbors | neighbors2)
            edge = (node, n)
            if edge not in graph.edges(node):
                tup = (edge, score)
                scores.append(tup)
    
    scores.sort(key = operator.itemgetter(1), reverse = True)
    for i in range(k):
        result.append(scores[i])
    
    return result        
        

def evaluate(predicted_edges, graph):

    result = 0
    predicted = 0
    
    for edge1 in predicted_edges:
        for edge2 in graph.edges():
            if edge1 == edge2:
                predicted = predicted +1
    
    result = predicted/len(predicted_edges)
    return result


def loadData(path,filename):
    fileroute = path+filename
    d = pickle.load( open( fileroute, "rb" ) )
    return d

def count_friends(users):
    cnt = Counter()
    for k,v in users.items():
        for f in users[k]['friends']:
            cnt[f] +=1
    return cnt
 

def create_graph(users, friend_counts):
    G = nx.Graph()
    common = friend_counts.most_common()
    
    for k in users.keys():
        G.add_node(k[1])
    for ele in common:
        if(ele[1]>1):
            G.add_node(ele[0])
            for k,v in users.items():
                if ele[0] in v:
                    G.add_edge(k[1], ele[0])
        else:
            break
    return G

def saveData(path, filename, s_object):
    fileroute = path+'/'+filename
    with open(fileroute, "wb") as f:
        pickle.dump(s_object, f, pickle.HIGHEST_PROTOCOL)
        
def id2ScreenName(data): 
    id2sn = []
    ids = []
    sn = []
    for k, v in data.items():
        id2sn.append((k,v['id']))
        ids.append(v['id'])
        sn.append(k)
    print(id2sn)
    return id2sn, ids, sn
        
def get_twitter():
    return TwitterAPI(consumer_key, consumer_secret, access_token, access_token_secret)

def robust_request(twitter, resource, params, max_tries=5):

    for i in range(max_tries):
        request = twitter.request(resource, params)
        if request.status_code == 200:
            return request
        else:
            print('Got error %s \nsleeping for 15 minutes.' % request.text)
            sys.stderr.flush()
            time.sleep(61 * 15) 

def getScreenName(twitter, userId):
    return robust_request(twitter, 'users/show', {'user_id': userId})   
    
def main():
    graph = load_graph() 
    twitter = get_twitter()

    print('graph has %d nodes and %d edges' %
          (graph.order(), graph.number_of_edges()))
    subgraph = get_subgraph(graph, 4)
    print('subgraph has %d nodes and %d edges' %
          (subgraph.order(), subgraph.number_of_edges()))
    print('norm_cut scores by max_depth:')
    print(score_max_depths(subgraph, range(1,3)))
    clusters = partition_girvan_newman(subgraph, 3)
    print('%d clusters' % len(clusters))
    print('first partition: cluster 1 has %d nodes and cluster 2 has %d nodes' %
          (clusters[0].order(), clusters[1].order()))
    print('smaller cluster nodes:')
    sn = []
    for node in sorted(clusters, key=lambda x: x.order())[0].nodes():
        sn.append(getScreenName(twitter, node).json()['name'])
    print(sn)
    
    data = loadData('./data/users/', 'users.file')
    id2sn, ids, sn = id2ScreenName(data)
    
    test_node = 818713465653051392

    train_graph = make_training_graph(subgraph, test_node, 5)
    print('train_graph has %d nodes and %d edges' %
          (train_graph.order(), train_graph.number_of_edges()))


    jaccard_scores = jaccard(train_graph, test_node, 5)
               
    print('\ntop jaccard scores for node '+ str(test_node)+':')
    print(jaccard_scores)

    
    data = dict()

    data['clusters']=(clusters[0].order(), clusters[1].order())
    saveData('./data', 'cluster.file', data)
       
if __name__ == '__main__':
    main()
    
    
    