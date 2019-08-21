#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Sun Apr 14 18:06:59 2019

@author: marina
"""

import collections
import random
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from matplotlib.pyplot import figure

import networkx as nx
import sys
import pickle
import os
import time
from sklearn.model_selection import KFold
from TwitterAPI import TwitterAPI



# Twitter keys
consumer_key = ''
consumer_secret = ''
access_token = ''
access_token_secret = ''

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
Retrieve the tweets for screen_name.
Params:
    twitter.......The TwitterAPI object.
    screen_name...A string of the user screen_name
    limit.........Maximum number of tweets retrieved
Returns:
    A list of tweets of the user
"""
def get_tweets(twitter, screen_name, limit):
    tweets = []
    request = robust_request(twitter, 'statuses/user_timeline', {'screen_name': screen_name, 'count': limit})
    for r in request:
        tweets.append(r['text'])
    return tweets

"""
Retrieve the list of friends for the user given
Params:
    twitter.......The TwitterAPI object.
    screen_name...A string of the user screen_name
Returns:
    A list of ids of the user's friends
"""
def get_friends(twitter, screen_name):
    friendids = []
    resources = 'friends/ids'
    request = robust_request(twitter, resources, {'screen_name':screen_name})
    friendids = sorted([id for id in request])
    return friendids

"""
Save a local file with the list of friends for each user
Params:
    twitter.......The TwitterAPI object.
    users.........A list of user objects
    path..........Path where to save the data
    filename......Name to give to the new created file
Returns:
     Nothing
"""
def add_all_friends(twitter, users,path,filename):
    for u in users:
        friends = get_friends(twitter, u['screen_name']) 
        u['friends']=friends
    saveData(path,filename, users)

"""
Print the number of friends per candidate, sorted by candidate name.
Args:
    users....The list of user dicts.
Returns:
     Nothing
"""
def print_num_friends(users):

    names = []
    cnt = Counter()
    for u in users:
        names.append(u['screen_name'])
        cnt[u['screen_name']] += len(u['friends']) 
    names = sorted(names)
    for n in names:
        print(str(n) +" has "+str(cnt[n])+" friends.")

"""
Count the number of friends for all candidates
Args:
    users....The list of user dicts.
Returns:
    A Counter object containing the number of friends for each user.
"""
def count_friends(users):
    cnt = Counter()
    for k,v in users.items():
        for f in users[k]['friends']:
            cnt[f] +=1
    return cnt

"""
Retrieve user objects for each screen_name
Args:
    twitter.......The TwitterAPI object.
    screen_names...A list of strings of the users screen_name
Returns:
     A list of user objects
"""       
def get_users(twitter, screen_names):
    users = []
    for name in screen_names:
        users += robust_request(twitter, 'users/lookup', {'screen_name': name})
    return users   

"""
Retrieve user screen names for each user id
Args:
    twitter.......The TwitterAPI object.
    userIds...A list of strings of the users ids
Returns:
     A list of user screen names
"""  
def getScreenName(twitter, userIds):
    screenNames = []     
    for userId in userIds:
        screenNames += robust_request(twitter, 'users/show', {'user_id': userId})
    return screenNames 

"""
Retrieve user id for each screen_name
Args:
    twitter.......The TwitterAPI object.
    screen_names...A list of strings of the users screen_name
Returns:
     A list of strings containing user ids
"""   
def getId(twitter, screen_name):
    resource = 'users/lookup'
    resp = robust_request(twitter, resource, {'screen_name':screen_name})
    userId = resp.json()[0]['id']
    return userId

"""
Retrieve user objects for each screen_name
Args:
    users...........A list containing user objects
    friend_counts...A Counter object containing the number of friends of each user
Returns:
     A graph representing the relationship between all users and friends.
"""  
def create_graph(users, friend_counts):
    G = nx.Graph()
    common = friend_counts.most_common()

    for k,v in users.items():
        G.add_node(v['id'])
    for ele in common:
        if(ele[1]>2):
            G.add_node(ele[0])
            for k,v in users.items():
                if ele[0] in users[k]['friends']:
                    G.add_edge(v['id'], ele[0])
        else:
            break
    return G

"""
Retrieve common friends between each pair of users
Args:
    users...........A list containing user objects
Returns:
    List containing each pair of users and the number of common friends, sorted by number of common friends.
""" 
def friend_overlap(users):
    overlap = []
    i = 0
    for u in users.keys():
        for u1 in users.keys():
            if(u != u1 and i < len(users)/2+1):
                common = set(users[u]['friends']).intersection(users[u1]['friends'])
                overlap.append((sorted([u, u1])[0], sorted([u, u1])[1],len(common)))
        i +=1
    overlap = list(set(overlap))
    sorted_name = sorted(overlap, key=lambda tup: tup[0])
    sorted_by_number = sorted(sorted_name, key=lambda tup: tup[2], reverse=True)
    return sorted_by_number

"""
Plot a network representing how the community of users is connected. Nodes are users and 
links represent connection between users.
Args:
    graph...........Nodes and edges between users
    users...........List of twitter user objects
    filename........Destination file to safe the figure
Returns:
     Nothing
""" 
def draw_network(graph, users, filename):
    
    pos= nx.spring_layout(graph)
    p = {}
    labels = {}
    i=0
    for u,v in users.items():
        labels[i]=u
        p[i]=pos[v['id']]
        i +=1
    nx.draw(graph, pos=pos, with_labels=False,node_size = 10,width=0.2)
    nx.draw_networkx_labels(graph, pos=p, font_color='w',labels=labels)
    figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')

    plt.savefig(filename)
    plt.show()

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
Load file data into a dictionary
Args:
    path...........Path where the file is saved
    filename.......Name of the file to load
Returns:
     Dictionary containing the data read from the file
"""
def loadData(path,filename):
    fileroute = path+filename
    d = pickle.load( open( fileroute, "rb" ) )
    return d    

"""
Retrieve the screen name from the user id
Args:
    twitter...........The TwitterAPI object
    users.............Name given to the file created
Returns:
     List containing the (screen_name,id) pair for each user
"""
def id2sn(twitter, users):
    id2sn = []
    
    for k,v in users.items():
        id2sn.append((k, v['id']))
        for f in v['friends']:
            sn = robust_request(twitter, 'users/show', {'user_id': f})
            id2sn.append((sn, f))
    return id2sn

"""
Main function, called upon execution of the program.
Retrieves the user data of a list of republicans and democrats.
Saves their user data into a local file named 'users.file' and 
the tweets into a file named 'test.file'.
Extracts the friends overlap between users and plots and saves a graph representing
the relationship between them.
Args:
    None
Returns:
    Nothing
"""
def main():
    
    dirData = "./data"
    usersPath = "./data/users"
    tweetsPath = "./data/tweets"
  
    if not (os.path.exists(dirData)):
        os.mkdir(dirData)
        
    twitter = get_twitter()
    
    if not (os.path.exists(usersPath)):
        os.mkdir(usersPath)

    # Getting republicans
    user1 = 'RepTomPrice'
    userId1 = getId(twitter, user1)
    user2 = 'GOPpolicy'
    userId2 = getId(twitter, user2)
    user3 = 'WaysandMeansGOP'
    userId3 = getId(twitter, user3)
    user4 = 'RosLehtinen'
    userId4 = getId(twitter, user4)
    user5 ='RobWittman'
    userId5 = getId(twitter, user5)

          
    # Getting democrats   
    user9 = 'RepDarrenSoto'
    userId9 = getId(twitter, user9)
    user10 = 'RepTomSuozzi '
    userId10 = getId(twitter, user10)
    user11 ='RepAlLawsonJr'
    userId11 = getId(twitter, user11)
    user12 ='RepEspaillat'
    userId12 = getId(twitter, user12)
    user13 ='RepBarragan'
    userId13 = getId(twitter, user13)

    republicans = []
    republicans.append((user1, userId1))
    republicans.append((user2, userId2))
    republicans.append((user3, userId3))
    republicans.append((user4, userId4))
    republicans.append((user5, userId5))
    
    democrats = []
    democrats.append((user9, userId9))
    democrats.append((user10, userId10))
    democrats.append((user11, userId11))
    democrats.append((user12, userId12))
    democrats.append((user13, userId13))
    
    users = dict()
    for user in republicans:
        users[user[0]] = {'Party': 'republican','id': user[1], 'friends':get_friends(twitter, user[0])}
        
    for user in democrats:
        users[user[0]] = {'Party': 'democrat','id': user[1], 'friends':get_friends(twitter, user[0])}

    if not (os.path.exists(usersPath)):
        os.mkdir(usersPath)
    saveData(usersPath, 'users.file', users)

     
        
    if not (os.path.exists(tweetsPath)):
        os.mkdir(tweetsPath) 
        
    tweets = dict()
    limit = 500
  
    # Label 1 for republicans, 0 for democrats. Useful for classify.py
    for k,v in users.items():
        tweets[k] = (v['Party'],get_tweets(twitter, k, limit))

    saveData(tweetsPath,'test.file',tweets)
    friend_counts = count_friends(users)

    print('Most common friends:\n%s' % str(friend_counts.most_common(5)))

    print('Friend Overlap:\n%s' % str(friend_overlap(users)))
            
    graph = create_graph(users, friend_counts)
    print('graph has %s nodes and %s edges' % (len(graph.nodes()), len(graph.edges())))
    print("CLOSE THE PLOT TO CONTINUE")
    draw_network(graph, users, 'network.png')
    print('network drawn to network.png')
    
    pickle.dump(graph, open('./data/graph.txt', 'wb'))
    
if __name__ == '__main__':
    main()