#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 12:41:17 2019

@author: marina
"""


import pickle


def saveData(path, filename, s_object):
    fileroute = path+'/'+filename
    with open(fileroute, "wb") as f:
        pickle.dump(s_object, f, pickle.HIGHEST_PROTOCOL)

def loadData(path,filename):
    fileroute = path+filename
    d = pickle.load( open( fileroute, "rb" ) )
    return d 
 
def load_graph():
    graph = pickle.load(open('./data/graph.txt','rb'))
    return graph

def readFile(path):
    data = pickle.load(open( path, "rb" ) )
    docs = []
    for k,v  in data.items():
        for d in v[1]:
            docs.append(d)

    return docs

def main():
    graph = load_graph()
    print('Total number of users collected: ',
          (graph.order()))   
    tweetsPath = "./data/tweets"
    data = loadData(tweetsPath,'/test.file')
    
    tweets = []
    for k,v  in data.items():
        for d in v[1]:
            tweets.append(d)
    print('Total number of tweets collected: ', len(tweets))
    clusterPath = "./data"
    data = loadData(clusterPath,'/cluster.file')
        
    print("Total number of clusters: ", len(data['clusters']))
    for i in range(len(data['clusters'])):
        print("Users in community "+str(i)+":"+ str(data['clusters'][i]))
    
   
    classPath = "./data"
    data1 = loadData(classPath,'/classif.file')
    print("Number of tweets classified as republicans: ",data1['rep'][0])
    print("Number of tweets classified as democrats: ",data1['dem'][0])
   
    print("Example of republican tweet: ",data1['rep'][1])
    print("Example of democrat tweet: ",data1['dem'][1])
    
    text_file = open("./Summary.txt", "w")
    text_file.write('Total number of users collected: '+ str(graph.order())+ '\n')
    text_file.write('Total number of tweets collected: '+ str(len(tweets))+ '\n')
    text_file.write("Total number of communities: "+ str(len(data['clusters'])) + '\n')
    text_file.write("Number of tweets classified as republicans: " + str(data1['rep'][0])+ '\n')
    text_file.write("Number of tweets classified as democrats: " + str(data1['dem'][0]) + '\n')
    text_file.write("Example of republican tweet: " + str(data1['rep'][1]) + '\n')
    text_file.write("Example of demodrat tweet: " + str(data1['dem'][1]) + '\n')   
    text_file.close()
   
if __name__ == '__main__':
    main()