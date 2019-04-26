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



# =============================================================================
# #YO NUEVAS
# consumer_key = 'yuBdTI32ESnZYdLuJAElEGvL5'
# consumer_secret = 'DNycDzVl3Yeqleog1DyiWMYKjBHjmo5S5kgw05yndMMyVfeJ35'
# access_token = '1090407968028459008-R9ZYN3WrmFSyOVtN4SGTVFtD75BI58'
# access_token_secret = 'OciUhBYBZp67foDHVMZuuirIC2aLGykhrsRjK3dCWngBK'
# =============================================================================
#JOSEPA
consumer_key = 'RRkJlRIsNGvsZ9E1GW8V5MRXP'
consumer_secret = 'jxmWkddzHgVSh5gsCqRPOUCBx5OCFdTAtMzhgqPdUIBNatDr7a'
access_token = '3020342014-gwUKeKgKGKTxP6xxFrRDec5Pwi6uE2gEZUlHie1'
access_token_secret = 'Vaqmg0cCa3hpQQT3zbApv2bzsiTZchbtX5xHDl8siqOMb'
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

def get_tweets(twitter, screen_name, limit):
    tweets = []
    request = robust_request(twitter, 'statuses/user_timeline', {'screen_name': screen_name, 'count': limit})
    for r in request:
        tweets.append(r['text'])
    return tweets

def get_friends(twitter, user):
    friendids = []
    resources = 'friends/ids'
    request = robust_request(twitter, resources, {'screen_name':user})
    friendids = sorted([id for id in request])
    return friendids


def add_all_friends(twitter, users,path,filename):

    for u in users:
        friends = get_friends(twitter, u['screen_name']) 
        u['friends']=friends
    saveData(path,filename, users)

def print_num_friends(users):
    """Print the number of friends per candidate, sorted by candidate name.
    See Log.txt for an example.
    Args:
        users....The list of user dicts.
    Returns:
        Nothing
    """
    names = []
    cnt = Counter()
    for u in users:
        names.append(u['screen_name'])
        cnt[u['screen_name']] += len(u['friends']) 
    names = sorted(names)
    for n in names:
        print(str(n) +" has "+str(cnt[n])+" friends.")

def count_friends(users):
    cnt = Counter()
    for k,v in users.items():
        for f in users[k]['friends']:
            cnt[f] +=1
    return cnt
        
def get_users(twitter, screen_names):
    users = []
    for name in screen_names:
        users += robust_request(twitter, 'users/lookup', {'screen_name': name})
    return users   

def getScreenName(twitter, userIds):
    screenNames = []     
    for userId in userIds:
        screenNames += robust_request(twitter, 'users/show', {'user_id': userId})
    return screenNames 
 
def getId(twitter, screen_name):
    resource = 'users/lookup'
    resp = robust_request(twitter, resource, {'screen_name':screen_name})
    userId = resp.json()[0]['id']
    return userId


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
    
def saveData(path, filename, s_object):
    fileroute = path+'/'+filename
    with open(fileroute, "wb") as f:
        pickle.dump(s_object, f, pickle.HIGHEST_PROTOCOL)

def loadData(path,filename):
    fileroute = path+filename
    d = pickle.load( open( fileroute, "rb" ) )
    return d    

def id2sn(twitter, users):
    id2sn = []
    
    for k,v in users.items():
        id2sn.append((k, v['id']))
        for f in v['friends']:
            sn = robust_request(twitter, 'users/show', {'user_id': f})
            id2sn.append((sn, f))
    return id2sn
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
# =============================================================================
#     user6 ='boblatta'
#     userId6 = getId(twitter, user6)
# =============================================================================
    
          
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
# =============================================================================
#     user14 ='RepMcEachin'
#     userId14 = getId(twitter, user14) 
# =============================================================================


    republicans = []
    republicans.append((user1, userId1))
    republicans.append((user2, userId2))
    republicans.append((user3, userId3))
    republicans.append((user4, userId4))
    republicans.append((user5, userId5))
  #  republicans.append((user6, userId6))

    
    democrats = []
    democrats.append((user9, userId9))
    democrats.append((user10, userId10))
    democrats.append((user11, userId11))
    democrats.append((user12, userId12))
    democrats.append((user13, userId13))
  #  democrats.append((user14, userId14))
    
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