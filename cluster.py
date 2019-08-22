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


consumer_key = ''
consumer_secret = ''
access_token = ''
access_token_secret = ''

"""
Create graph from text file data
Params:
    None
Returns:
    Graph object with the data read from the text file
"""
def load_graph():
    graph = pickle.load(open('./data/graph.txt','rb'))
    return graph

 """
Implement Breadth-First Search algorithm. 
Iterates over a queue of nodes. Starts from the root node and adds its neighbors to the queue. 
The algorithm will stop when the maximum depth is reached or when all the nodes have been visited.
Creates a list of seen nodes to avoid visitting a node twice.
Params:
    graph.......A networkx Graph
    root........The root node in the search graph (a string). We are computing
                shortest paths from this node to all others.
    max_depth...An integer representing the maximum depth to search.
Returns:
    node2distances...dict from each node to the length of the shortest path from
                    the root node
    node2num_paths...dict from each node to the number of shortest paths from the
                    root node to this node.
    node2parents.....dict from each node to the list of its parents in the search
                    tree
"""   
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
      

 """
 Compute the final step of the Girvan-Newman algorithm.
 Params:
      root.............The root node in the search graph (a string). We are computing
                       shortest paths from this node to all others.
      node2distances...dict from each node to the length of the shortest path from
                       the root node
      node2num_paths...dict from each node to the number of shortest paths from the
                       root node that pass through this node.
      node2parents.....dict from each node to the list of its parents in the search
                       tree
    Returns:
      A dict mapping edges to credit value. Each key is a tuple of two strings
      representing an edge (e.g., ('A', 'B')). Each tuple is sorted alphabetically
 """     
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
   
"""
Compute the approximate betweenness of each edge, using max_depth to reduce
computation time in breadth-first search. bfs and bottom_up functions are called for each node
in the graph

Params:
    graph.......A networkx Graph
    max_depth...An integer representing the maximum depth to search.

Returns:
    A dict mapping edges to betweenness. Each key is a tuple of two strings
    representing an edge (e.g., ('A', 'B')). Tuples are sorted alphabetically
"""   
def approximate_betweenness(graph, max_depth):

    betweenness = defaultdict(float)
    
    
    for node1 in graph.nodes():
        node2distances, node2num_paths, node2parents = bfs(graph, node1, max_depth)
        credit = bottom_up(node1, node2distances, node2num_paths, node2parents)
        for e, b in credit.items():
            betweenness[e] += b/2
            

    return betweenness

"""
A helper function that returns the list of all connected components in the given graph.
"""
def get_components(graph):

    return [c for c in nx.connected_component_subgraphs(graph)]


"""
Make a partition of the graph using the implemented method approximate_betweenness.
The method computes the approximate betweennes of all edges, and removes edges until 
more than one component is created and returns the components.
Params:
    graph.......A networkx Graph
    max_depth...An integer representing the maximum depth to search.

Returns:
    A list of networkx Graph objects, one per partition.
"""
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

"""
Return a subgraph containing nodes whose degree is greater than or equal to min_degree.
This function will be used in the main method to prune the original graph.
Params:
    graph........a networkx graph
    min_degree...degree threshold
Returns:
    a networkx graph, filtered as defined above.
""" 
def get_subgraph(graph, min_degree):

    subgraph = graph.copy()
    
    for node in graph.nodes():
        degree = graph.degree[node]
        if degree < min_degree:
            subgraph.remove_node(node)
            
    return subgraph

"""
Compute the volume for a list of nodes, which is the number of edges in `graph` with at least one end in nodes.
Params:
    nodes...a list of strings for the nodes to compute the volume of.
    graph...a networkx graph
Return:
    volume: int with the number of edges with one end in nodes
"""
def volume(nodes, graph):
    volume = 0
    for edge in graph.edges():
        for n in nodes:
            if(edge[0] == n or edge[1] == n):
                volume += 1
                break
    
    return volume
                

"""
Compute the cut-set of the cut (S,T), which is
the set of edges that have one endpoint in S and
the other in T.
Params:
    S.......set of nodes in first subset
    T.......set of nodes in second subset
    graph...networkx graph
Returns:
    An int representing the cut-set.
"""
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
            
 """
    The normalized cut value for the cut S/T. 
    Params:
      S.......set of nodes in first subset
      T.......set of nodes in second subset
      graph...networkx graph
    Returns:
      A float representing the normalized cut value
"""           
def norm_cut(S, T, graph):
    ncv = 0
    
    cutST = cut(S, T, graph)
    volS = volume(S, graph)
    
    volT = volume(T, graph)
    ncv = (cutST/volS) + (cutST/volT)
    return ncv

 """
Enumerate over all possible cuts of the graph, up to max_size, and compute the norm cut score.
Params:
    graph......graph to be partitioned
    max_size...maximum number of edges to consider for each cut.
                E.g, if max_size=2, consider removing edge sets
                of size 1 or 2 edges.
Returns:
    (unsorted) list of (score, edge_list) tuples, where
    score is the norm_cut score for each cut, and edge_list
    is the list of edges (source, target) for each cut.
"""   
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
    
"""
In order to assess the quality of the approximate partitioning method
we've developed, we will run it with different values for max_depth
and see how it affects the norm_cut score of the resulting partitions.
Recall that smaller norm_cut scores correspond to better partitions.

Params:
    graph........a networkx Graph
    max_depths...a list of ints for the max_depth values to be passed
                to calls to partition_girvan_newman

Returns:
    A list of (int, float) tuples representing the max_depth and the
    norm_cut value obtained by the partitions returned by
    partition_girvan_newman.
    """    
def score_max_depths(graph, max_depths):

    result = []
    max_depth = 0
    
    for md in max_depths:
       max_depth = md
       partitions = partition_girvan_newman(graph, max_depth)
       ncv = norm_cut(partitions[0],partitions[1],graph)
       result.append((max_depth,ncv))
       
    return result


"""
    Remove the edges to the first n neighbors of
    test_node, where the neighbors are sorted alphabetically.

    Params:
      graph.......a networkx Graph
      test_node...a string representing one node in the graph whose
                  edges will be removed.
      n...........the number of edges to remove.

    Returns:
      A *new* networkx Graph with n edges removed.
    """
def make_training_graph(graph, test_node, n):

    graphCopy = graph.copy()
    neigh = sorted(graph.neighbors(test_node))[:n]
    for n in neigh:
      graphCopy.remove_edge(test_node, n)
    return graphCopy
        
 """
Compute the k highest scoring edges to add to this node based on
the Jaccard similarity measure.
Note that we don't return scores for edges that already appear in the graph.

Params:
    graph....a networkx graph
    node.....a node in the graph (a string) to recommend links for.
    k........the number of links to recommend.

Returns:
    A list of tuples in descending order of score representing the
    recommended new edges. Ties are broken by
    alphabetical order of the terminal node in the edge.
"""      
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
        
"""
Return the fraction of the predicted edges that exist in the graph.

Args:
    predicted_edges...a list of edges (tuples) that are predicted to
                    exist in this graph
    graph.............a networkx Graph
"""
def evaluate(predicted_edges, graph):

    result = 0
    predicted = 0
    
    for edge1 in predicted_edges:
        for edge2 in graph.edges():
            if edge1 == edge2:
                predicted = predicted +1
    
    result = predicted/len(predicted_edges)
    return result

"""
Load the data from Twitter saved when running collect.py
Args:
    path...............path where the file is saved
    filename...........name of the file
Return:
    dict containing the data from Twitter
"""
def loadData(path,filename):
    fileroute = path+filename
    d = pickle.load( open( fileroute, "rb" ) )
    return d

"""
Save into a local file the data retrieved from Twitter
Args:
    path...........Path where the file will be saved
    filename.......Name given to the file created
    s_object.......Dict containing the data to save
Returns:
     Nothing
"""  
def saveData(path, filename, s_object):
    fileroute = path+'/'+filename
    with open(fileroute, "wb") as f:
        pickle.dump(s_object, f, pickle.HIGHEST_PROTOCOL)

"""
Retrieve the screen name from the user id
Args:
    data...........Twitter user data
Returns:
    id2sn: List of tuples, containing each tuple the screen name and id 
    ids: List of strings, each string is a user id
    sn: List of strings, each string is a screen name
"""        
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

"""
Construct an instance of TwitterAPI using the tokens entered above.
Returns:
    An instance of TwitterAPI.
"""       
def get_twitter():
    return TwitterAPI(consumer_key, consumer_secret, access_token, access_token_secret)


"""
Handle Twitter's rate limiting
Sleep for 15 minutes if a Twitter request fails.
Do this at most max_tries times before quitting.
    Args:
      twitter .... A TwitterAPI object.
      resource ... A resource string to request; e.g., "friends/ids"
      params ..... A parameter dict for the request, e.g., to specify
                   parameters like screen_name or count.
      max_tries .. The maximum number of tries to attempt.
    Returns:
      A TwitterResponse object, or None if failed.
"""
def robust_request(twitter, resource, params, max_tries=5):

    for i in range(max_tries):
        request = twitter.request(resource, params)
        if request.status_code == 200:
            return request
        else:
            print('Got error %s \nsleeping for 15 minutes.' % request.text)
            sys.stderr.flush()
            time.sleep(61 * 15) 


"""
Retrieve user screen names for each user id
Args:
    twitter.......The TwitterAPI object.
    userIds...A list of strings of the users ids
Returns:
     A list of user screen names
""" 
def getScreenName(twitter, userId):
    return robust_request(twitter, 'users/show', {'user_id': userId})   
    
"""
Main function, called upon execution of the program.
Creates a graph with the Twitter data collected from collect.py
Calculates the number of clusters using the Girvan-Newman algorithm
Prints the number of clusters and its nodes
Creates a new training graph from one of the subgraphs and prints the graph information
Computes the jaccard similarity measure for the training node
Saves the data of the clusters into a local file.
Args:
    None
Returns:
    Nothing
"""
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
    
    
    