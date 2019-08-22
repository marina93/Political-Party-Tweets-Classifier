#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 12:41:17 2019

@author: marina
"""

"""
This function uses the data extracted in the previous classes (collect.py, cluster.py and classify.py) 
and plots and saves into a file named 'Summarize.txt' the main results and conclusion extracted.
"""
import pickle

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
Load tweets from republicans and democrats saved in a local file
Args:
    path...........path to local file
Return:
    docs: List of strings, one for each tweet from the data file
    labels: List of strings, each representing the label from the data item ('0' for democrat and '1' for republican)
"""
def readFile(path):
    data = pickle.load(open( path, "rb" ) )
    docs = []
    for k,v  in data.items():
        for d in v[1]:
            docs.append(d)

    return docs

"""
Main function, called upon execution of the program.
Shows on the console the number of users and tweets collected, together with the 
data extracted when executing cluster.py and classify.py. Shows the number of communitties
and the number of users that each community has.
Creates a file 'summary.txt' and writes to it the main details of the data used and the results obtained.
Args:
    None
Returns:
    Nothing
"""
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