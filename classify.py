#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 12:41:17 2019

@author: marina
"""


import itertools
from collections import Counter, defaultdict
from itertools import chain, combinations
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import csv
import pickle
import re
import string
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
import string
import tarfile
import codecs
import zipfile
import urllib.request
import collections

from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize

""" 
This class uses a Logistic Regresion model to classify the tweets collected in collect.py into two categories: Democrats and republicans. 
To this aim,it uses an external dataset with 84.000 tweets already labelled for training the model.
"""
def download_data():

    url = 'https://www.dropbox.com/s/qmopda2wl0zo0hv/democratvsrepublicantweets.zip?dl=1'
    urllib.request.urlretrieve(url, 'train.zip')
    zfile = zipfile.ZipFile('train.zip')
    zfile.extractall('./data/tweets/')
    zfile.close()

def loadData(path):
    d = pickle.load( open( path, "rb" ) )
    data = []
    for k, v in d.items():
        if(v[0] == 'republican'):
            iD = 1
        elif(v[0] == 'democrat'):
            iD = 0
        data.append([iD, v[1]])
    return np.array([d[1] for d in data]), np.array([d[0] for d in data])

# Label 1 f or Republican, 0 for Democrats
def readCSV(path):
    data = []

    with open(path, 'r') as csvFile:
        reader = csv.DictReader(csvFile)
        
        for row in reader:
            if(row['Party'] == 'Republican'):
                iD = 1
            elif(row['Party'] == 'Democrat'): 
                iD = 0
            
            if(row['Handle']!='RepTomPrice' and row['Handle']!='GOPpolicy' and row['Handle']!='WaysandMeansGOP' and row['Handle']!='RosLehtinen' and row['Handle']!='RobWittman'
               or row['Handle']!='RepDarrenSoto' and row['Handle']!='CongressmanRaja ‏' and row['Handle']!='RepAlLawsonJr' and row['Handle']!='RepEspaillat' and row['Handle']!='RepMcEachin'):
                    data.append([iD, row['Tweet']])
      
    return np.array([d[1] for d in data]), np.array([d[0] for d in data])


def readFile(path):
    data = pickle.load(open( path, "rb" ) )
    docs = []
    labels = []
    for k,v  in data.items():
        for d in v[1]:
            if v[0] == 'republican':
                labels.append(1)
            elif v[0] == 'democrat':
                labels.append(0)
            docs.append(d)

    return docs, labels

def tokenize(doc, keep_internal_punct=False):

    tokens = []
    doc = ''.join([x for x in doc if x in string.ascii_letters + string.digits + '\'- '])
    
    if not doc:
        return []
    tokens = []
    if keep_internal_punct == False:
        tokens = re.sub('\W+', ' ', doc).lower().split()
    else: 
        tokens = doc.lower().split()
      
    stop_words = set(stopwords.words('english')) 
    filtered_sentence = []
    for w in tokens: 
        if w not in stop_words: 
            filtered_sentence.append(w) 
      
    return filtered_sentence
        
def token_features(tokens, feats):

    for token, count in collections.Counter(tokens).items():
        st = 'token='+token
        feats[st]=count
        
def token_pair_features(tokens, feats, k=3):

    subArrays = []
    for i in range(len(tokens)):
        if i+k <= len(tokens):
            subArrays.append(tokens[i:i+k])
        elif(i-len(tokens)>0): 
            subArrays.append(tokens[i:len(tokens)])
    
    
    st = ""
    
    for subArray in subArrays:   
        tokens = []
        for i in range(len(subArray)):
            for j in range(len(subArray)):
                if i!=j:
                    if i < j:
                        st="token_pair="+subArray[i]+"__"+subArray[j]
                    else: 
                        st="token_pair="+subArray[j]+"__"+subArray[i]
                if st not in tokens:
                    tokens.append(st)
        for token in tokens:
            feats[token]+=1

rep_words = set(['God','benghazi', 'Psalm', 'America', 'border', 'Obama', 'Obamacare','Reid', 'Pelosi','democrats'])
dem_words = set(['lol', 'happy', 'like', 'amazing', 'swear','republicans','Trump','wall'])

def lexicon_features(tokens, feats):

    tokens2 = []
    rep_words2 = []
    dem_words2 = []
    for token in tokens:
        token = token.lower()
        tokens2.append(token)
        
    for rep_word in rep_words:
        rep_word = rep_word.lower()
        rep_words2.append(rep_word)
        
    for dem_word in dem_words:
        dem_word = dem_word.lower()
        dem_words2.append(dem_word)
        
    rep = list(set(tokens2).intersection(rep_words2))
    dem = list(set(tokens2).intersection(dem_words2))
    
    feats['rep_words'] = len(rep)
    feats['dem_words'] = len(dem)

def featurize(tokens, feature_fns):

    result = []
    for feat in list(feature_fns):
        feats = defaultdict(lambda: 0)
        feat(tokens,feats)
        for f in feats.items():
            result.append(f)

    result = sorted(result)
    return result  
    
def create_feat_array(tokens_doc, tokens, feats):

    
    feats1 = []
     
    for i in range(len(tokens_doc)):
        if len(tokens) > len(tokens_doc[i]):
            A = tokens
            B = tokens_doc[i]
        else:
            B = tokens
            A = tokens_doc[i]
        diff = set(A).difference(set(B))
        ap = []
        for d in diff:
            st = "token="+d
            ap.append(st)
            feats[i].append((st,0))
        feats1.append(sorted(feats[i]))
     
    return feats1
        
   
def vectorize(tokens_list, feature_fns, min_freq, vocab=None):

    
    feats = []
    vals = []
    col_idx = []
    row_ptr = [0]
    featDict = defaultdict(lambda:0)
    
       
    
    for doc in tokens_list:
        feats_doc = featurize(doc, feature_fns)
        
        feats.append(feats_doc)
        for feat in feats_doc:
            featDict[feat] += 1
    
    featDict = featDict.items()
    featDict = sorted([x[0][0] for x in featDict if x[1]>=min_freq])

    if vocab == None:
       vocab = dict.fromkeys(d for d in featDict)
       vocab.update((k, i) for i, k in enumerate(vocab))

    for d in feats:
        for feat, v in d:
            if feat in vocab:
                col_idx.append(vocab[feat])
                vals.append(v)
        row_ptr.append(len(col_idx))
    
    X = csr_matrix((vals, col_idx, row_ptr), dtype = np.int64, shape = (len(tokens_list),len(vocab)))
    return X, vocab


def accuracy_score(truth, predicted):

    return len(np.where(truth==predicted)[0]) / len(truth)



def cross_validation_accuracy(clf, X, labels, k):

    cv = KFold(n_splits=k)
    accuracies = []
    for train_idx, test_idx in cv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]
        clf.fit(X_train, y_train)
        predicted = clf.predict(X_test)
        acc = accuracy_score(y_test, predicted)
        accuracies.append(acc)
    avg = np.mean(accuracies)
    return avg

 
def eval_all_combinations(docs, labels, punct_vals,feature_fns, min_freqs):

    results = []
    tokens_list = []
    
    comb = []
    
    comb = list(chain(*map(lambda x: combinations(feature_fns, x), range(1, len(feature_fns)+1))))
    for punct in punct_vals:
        tokens_list = []
        
        for doc in docs:
            tokens = tokenize(doc,punct)
            tokens_list.append(tokens)
            
        for i in range(len(comb)):
                for min_freq in min_freqs:
                    c = comb[i]
                    X, vocab = vectorize(tokens_list,c,min_freq)
                    clf = LogisticRegression()
                    accuracy = cross_validation_accuracy(clf,X,labels,5) 
                    results.append({'punct':punct,'features':c,'min_freq':min_freq,'accuracy':accuracy})
    results = sorted(results, key=lambda k: k['accuracy'], reverse=True)               
    return results
    
def plot_sorted_accuracies(results):
  
    y = []
    
    results = sorted(results, key=lambda k: k['accuracy'], reverse=False) 
    
  
    for r in results:
        y.append(r["accuracy"])
    x = np.arange(len(y))
    
    g = plt.figure()
    plt.plot(x,y)
    plt.title("Accuracy")
    plt.show()
    g.savefig('accuracies.png')

def mean_accuracy_per_setting(results):
    
    
    possiblePunct = []
    possibleFeats = []
    possibleFreqs = []
    possibleSettings = []
    meanAcc = []
    num = defaultdict(lambda: 0)
    den = defaultdict(lambda: 0)
    
    
    
    for result in results:

        possiblePunct.append(result['punct'])
        possibleFeats.append(result['features'])
        possibleFreqs.append(result['min_freq'])
        
        possibleSettings.append(('punct', result['punct']))
        possibleSettings.append(('features',result['features']))
        possibleSettings.append(('min_freq', result['min_freq']))
    
    possiblePunct = set(possiblePunct)
    possibleFeats = set(possibleFeats)
    possibleFreqs = set(possibleFreqs)
    possibleSettings = set(possibleSettings)
         
    for s, v in possibleSettings:

        for result in results:

            if(result[s]==v):
                num[v]+= result['accuracy']
                den[v]+=1 
        acc = num[v]/den[v]
        tup = (acc,v)
        meanAcc.append(tup)

         
    meanAcc = sorted(meanAcc, key=lambda x: x[0], reverse = True)
    return meanAcc


def fit_best_classifier(docs, labels, best_result):

    punct = best_result['punct']
    feature_fns = best_result['features']
    min_freq = best_result['min_freq']
    tokens_list = []
    cv = KFold(n_splits=5)
    clf = LogisticRegression()
    feat_names = []
    vocab = {}
    
    for doc in docs:
        tokens_list.append(tokenize(doc, punct))
                    
    X, vocab = vectorize(tokens_list, feature_fns, min_freq, vocab=None)
    for train_idx, test_idx in cv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]
        
        clf.fit(X_train, y_train)
       
    i=0
    for name in feat_names:             
            vocab[name] = i
            i+=1
            
    return clf, vocab  

def top_coefs(clf, label, n, vocab):

    result = []
    c = clf.coef_[0]
    indx = []

    if label == 1:
      indx = np.argsort(c)[::-1][:n]
    elif label == 0:
      indx = np.argsort(c)[:n]

    for i in indx:
      for k, v in vocab.items():
        if v == i:
          result.append((k,c[i]))
          
    return result

def parse_test_data(best_result, vocab):

    test_docs, test_labels = readFile(os.path.join('./data/tweets', 'test.file'))
    print("len test docs: ",len(test_docs))
    punct = best_result['punct']
    feature_fns = best_result['features']
    min_freq = best_result['min_freq']
    tokens_list = []
    for doc in test_docs:
      tokens_list.append(tokenize(doc, punct)) 
    X_test,_ = vectorize(tokens_list, feature_fns, min_freq, vocab)
    
    return test_docs, test_labels, X_test
    
def print_top_misclassified(test_docs, test_labels, X_test, clf, n):

    probs = clf.predict_proba(X_test)
    predicted = clf.predict(X_test)
    indx =[]
    
    for i in range(len(test_labels)):
        if predicted[i]!=test_labels[i]:
            if predicted[i]==0:
                indx.append((i,probs[i][0]))
            else:
                indx.append((i,probs[i][1]))

    sorted_index = sorted(indx, key = lambda k:k[1], reverse = True)[:n]
    for s in sorted_index:
        index = s[0]    

def saveData(path, filename, s_object):
    fileroute = path+'/'+filename
    with open(fileroute, "wb") as f:
        pickle.dump(s_object, f, pickle.HIGHEST_PROTOCOL)
        
def main():

    
    download_data()

    feature_fns = [token_features, token_pair_features, lexicon_features]
    download_data()
    docs1, labels1 = readCSV(os.path.join('./data/tweets', 'ExtractedTweets.csv'))
    docs = []
    labels = []
    # PODRÍA ESTAR FALLANDO ESTO?
    docs.extend(docs1[0:10000])
    docs.extend(docs1[-10000:-1])
    docs = np.asarray(docs)
    labels.extend(labels1[0:10000])
    labels.extend(labels1[-10000:-1])
    labels = np.asarray(labels)


    results = eval_all_combinations(docs, labels, [True, False],feature_fns,[2,5,10])
    best_result = results[0]
    worst_result = results[-1]
    print('best cross-validation result:\n%s' % str(best_result))
    print('worst cross-validation result:\n%s' % str(worst_result))
    #plot_sorted_accuracies(results)
    print('\nMean Accuracies per Setting:')
    print('\n'.join(['%s: %.5f' % (s,v) for v,s in mean_accuracy_per_setting(results)]))
    # Fit best classifier.
    clf, vocab = fit_best_classifier(docs, labels, results[0])
    print('\nTOP COEFFICIENTS PER CLASS:')
    print('Democrats words:')
    print('\n'.join(['%s: %.5f' % (t,v) for t,v in top_coefs(clf, 0, 5, vocab)]))
    print('\Republicans words:')
    print('\n'.join(['%s: %.5f' % (t,v) for t,v in top_coefs(clf, 1, 5, vocab)]))
    test_docs, test_labels, X_test = parse_test_data(best_result, vocab)
    
    predictions = clf.predict(X_test)
    rep0 = [a for a in predictions if a == 1]    
    dem0  = [a for a in predictions if a == 0]
    
    rep = test_labels.index(1)
    dem = test_labels.index(0)
           
    classif = dict()
    classif['rep'] = (len(rep0), test_docs[rep])
    classif['dem'] = (len(dem0), test_docs[dem])
    saveData('./data','/classif.file',classif)

    print('testing accuracy=%f' % accuracy_score(test_labels, predictions))
    

if __name__ == '__main__':
    main()