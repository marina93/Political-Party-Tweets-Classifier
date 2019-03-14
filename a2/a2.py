#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 12:41:17 2019

@author: marina
"""

# coding: utf-8

"""
CS579: Assignment 2
In this assignment, you will build a text classifier to determine whether a
movie review is expressing positive or negative sentiment. The data come from
the website IMDB.com.
You'll write code to preprocess the data in different ways (creating different
features), then compare the cross-validation accuracy of each approach. Then,
you'll compute accuracy on a test set and do some analysis of the errors.
The main method takes about 40 seconds for me to run on my laptop. Places to
check for inefficiency include the vectorize function and the
eval_all_combinations function.
Complete the 14 methods below, indicated by TODO.
As usual, completing one method at a time, and debugging with doctests, should
help.
"""

# No imports allowed besides these.
import itertools
from collections import Counter, defaultdict
from itertools import chain, combinations
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import string
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
import string
import tarfile
import urllib.request
import collections



def download_data():
    """ Download and unzip data.
    DONE ALREADY.
    """
    url = 'https://www.dropbox.com/s/8oehplrobcgi9cq/imdb.tgz?dl=1'
    urllib.request.urlretrieve(url, 'imdb.tgz')
    tar = tarfile.open("imdb.tgz")
    tar.extractall()
    tar.close()


def read_data(path):
    """
    Walks all subdirectories of this path and reads all
    the text files and labels.
    DONE ALREADY.
    Params:
      path....path to files
    Returns:
      docs.....list of strings, one per document
      labels...list of ints, 1=positive, 0=negative label.
               Inferred from file path (i.e., if it contains
               'pos', it is 1, else 0)
    """
    fnames = sorted([f for f in glob.glob(os.path.join(path, 'pos', '*.txt'))])
    data = [(1, open(f).readlines()[0]) for f in sorted(fnames)]
    fnames = sorted([f for f in glob.glob(os.path.join(path, 'neg', '*.txt'))])
    data += [(0, open(f).readlines()[0]) for f in sorted(fnames)]
    data = sorted(data, key=lambda x: x[1])
    return np.array([d[1] for d in data]), np.array([d[0] for d in data])


def tokenize(doc, keep_internal_punct=False):
    """
    Tokenize a string.
    The string should be converted to lowercase.
    If keep_internal_punct is False, then return only the alphanumerics (letters, numbers and underscore).
    If keep_internal_punct is True, then also retain punctuation that
    is inside of a word. E.g., in the example below, the token "isn't"
    is maintained when keep_internal_punct=True; otherwise, it is
    split into "isn" and "t" tokens.
    Params:
      doc....a string.
      keep_internal_punct...see above
    Returns:
      a numpy array containing the resulting tokens.
    >>> tokenize(" Hi there! Isn't this fun?", keep_internal_punct=False)
    array(['hi', 'there', 'isn', 't', 'this', 'fun'], dtype='<U5')
    >>> tokenize("Hi there! Isn't this fun? ", keep_internal_punct=True)
    array(['hi', 'there', "isn't", 'this', 'fun'], dtype='<U5')
    """
    tokens = []
   # result = doc.split()
    
    doc = ''.join([x for x in doc if x in string.ascii_letters + string.digits + '\'- '])
    
    if not doc:
        return []
    tokens = []
    if keep_internal_punct == False:
        tokens = re.sub('\W+', ' ', doc).lower().split()
    else: 
        tokens = doc.lower().split()
    return tokens
        
def token_features(tokens, feats):
    """
    Add features for each token. The feature name
    is pre-pended with the string "token=".
    Note that the feats dict is modified in place,
    so there is no return value.
    Params:
      tokens...array of token strings from a document.
      feats....dict from feature name to frequency
    Returns:
      nothing; feats is modified in place.
    >>> feats = defaultdict(lambda: 0)
    >>> token_features(['hi', 'there', 'hi'], feats)
    >>> sorted(feats.items())
    [('token=hi', 2), ('token=there', 1)]
    """
    for token, count in collections.Counter(tokens).items():
        st = 'token='+token
        feats[st]=count
    #print("feats", feats)
        

def token_pair_features(tokens, feats, k=3):
    """
    Compute features indicating that two words occur near
    each other within a window of size k.
    For example [a, b, c, d] with k=3 will consider the
    windows: [a,b,c], [b,c,d]. In the first window,
    a_b, a_c, and b_c appear; in the second window,
    b_c, c_d, and b_d appear. This example is in the
    doctest below.
    Note that the order of the tokens in the feature name
    matches the order in which they appear in the document.
    (e.g., a__b, not b__a)
    Params:
      tokens....array of token strings from a document.
      feats.....a dict from feature to value
      k.........the window size (3 by default)
    Returns:
      nothing; feats is modified in place.
    >>> feats = defaultdict(lambda: 0)
    >>> token_pair_features(np.array(['a', 'b', 'c', 'd']), feats)
    >>> sorted(feats.items())
    [('token_pair=a__b', 1), ('token_pair=a__c', 1), ('token_pair=b__c', 2), ('token_pair=b__d', 1), ('token_pair=c__d', 1)]
    """
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


neg_words = set(['bad', 'hate', 'horrible', 'worst', 'boring'])
pos_words = set(['awesome', 'amazing', 'best', 'good', 'great', 'love', 'wonderful'])

def lexicon_features(tokens, feats):
    """
    Add features indicating how many times a token appears that matches either
    the neg_words or pos_words (defined above). The matching should ignore
    case.
    Params:
      tokens...array of token strings from a document.
      feats....dict from feature name to frequency
    Returns:
      nothing; feats is modified in place.
    In this example, 'LOVE' and 'great' match the pos_words,
    and 'boring' matches the neg_words list.
    >>> feats = defaultdict(lambda: 0)
    >>> lexicon_features(np.array(['i', 'LOVE', 'this', 'great', 'boring', 'movie']), feats)
    >>> sorted(feats.items())
    [('neg_words', 1), ('pos_words', 2)]
    """
    tokens2 = []
    neg_words2 = []
    pos_words2 = []
    for token in tokens:
        token = token.lower()
        tokens2.append(token)
        
    for neg_word in neg_words:
        neg_word = neg_word.lower()
        neg_words2.append(neg_word)
        
    for pos_word in pos_words:
        pos_word = pos_word.lower()
        pos_words2.append(pos_word)
        
    neg = list(set(tokens2).intersection(neg_words2))
    pos = list(set(tokens2).intersection(pos_words2))
    
    feats['neg_words'] = len(neg)
    feats['pos_words'] = len(pos)



def featurize(tokens, feature_fns):
    """
    Compute all features for a list of tokens from
    a single document.
    Params:
      tokens........array of token strings from a document.
      feature_fns...a list of functions, one per feature
    Returns:
      list of (feature, value) tuples, SORTED alphabetically
      by the feature name.
    >>> feats = featurize(np.array(['i', 'LOVE', 'this', 'great', 'movie']), [token_features, lexicon_features])
    >>> feats
    [('neg_words', 0), ('pos_words', 2), ('token=LOVE', 1), ('token=great', 1), ('token=i', 1), ('token=movie', 1), ('token=this', 1)]
    """
    result = []
    #if(type(feature_fns) == "tuple"):
    for feat in list(feature_fns):
        feats = defaultdict(lambda: 0)
        feat(tokens,feats)
        for f in feats.items():
            result.append(f)
# =============================================================================
#     else:
#         feats = defaultdict(lambda: 0)
#         feature_fns(tokens,feats)
#         for f in feats.items():
#             result.append(f)
# =============================================================================
    
    result = sorted(result)
    return result  
    
def create_feat_array(tokens_doc, tokens, feats):

    
    # Tokens of "tokens array" that do not exist in "doc i"
    feats1 = []
     
    # Extract tokens that do not appear in doc
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
    
    """
    Given the tokens for a set of documents, create a sparse
    feature matrix, where each row represents a document, and
    each column represents a feature.
    Params:
      tokens_list...a list of lists; each sublist is an
                    array of token strings from a document.
      feature_fns...a list of functions, one per feature
      min_freq......Remove features that do not appear in
                    at least min_freq different documents.
    Returns:
      - a csr_matrix: See https://goo.gl/f5TiF1 for documentation.
      This is a sparse matrix (zero values are not stored).
      - vocab: a dict from feature name to column index. NOTE
      that the columns are sorted alphabetically (so, the feature
      "token=great" is column 0 and "token=horrible" is column 1
      because "great" < "horrible" alphabetically),
    When vocab is None, we build a new vocabulary from the given data.
    when vocab is not None, we do not build a new vocab, and we do not
    add any new terms to the vocabulary. This setting is to be used
    at test time.
    >>> docs = ["Isn't this movie great?", "Horrible, horrible movie"]
    >>> tokens_list = [tokenize(d) for d in docs]
    >>> feature_fns = [token_features]
    >>> X, vocab = vectorize(tokens_list, feature_fns, min_freq=1)
    >>> type(X)
    <class 'scipy.sparse.csr.csr_matrix'>
    >>> X.toarray()
    array([[1, 0, 1, 1, 1, 1],
           [0, 2, 0, 1, 0, 0]], dtype=int64)
    >>> sorted(vocab.items(), key=lambda x: x[1])
    [('token=great', 0), ('token=horrible', 1), ('token=isn', 2), ('token=movie', 3), ('token=t', 4), ('token=this', 5)]
    """
    # 1d array: All tokens of all docs
    tokens = []  
    
    # 2d array: Tokens of each doc
    tokens_doc = []
    
    # 2d array: Feats of each doc
    feats = []
    
    # Dict that matches feat to col
    
    
    # Name of all features
    feat_names = []
    
    # Dict that matches each token with the number of documents in which it appears
    count= {}
    vals = []
    col_idx = []
    row_ptr = [0]
    featDict = defaultdict(lambda:0)
    
       
    # Creates 2d array of features, where each row is a doc, columns are features.
    
    for doc in tokens_list:
        feats_doc = featurize(doc, feature_fns)
        
        feats.append(feats_doc)
        for feat in feats_doc:
            featDict[feat] += 1
    
    featDict = featDict.items()
    featDict = sorted([x[0][0] for x in featDict if x[1]>=min_freq])
   # featDict = sorted([x[0][0] for x in featDict if x[1]>=min_freq])    

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

    return vector, vocab

def accuracy_score(truth, predicted):
    """ Compute accuracy of predictions.
    DONE ALREADY
    Params:
      truth.......array of true labels (0 or 1)
      predicted...array of predicted labels (0 or 1)
    """
    return len(np.where(truth==predicted)[0]) / len(truth)



def cross_validation_accuracy(clf, X, labels, k):
    """
    Compute the average testing accuracy over k folds of cross-validation. You
    can use sklearn's KFold class here (no random seed, and no shuffling
    needed).
    Params:
      clf......A LogisticRegression classifier.
      X........A csr_matrix of features.
      labels...The true labels for each instance in X
      k........The number of cross-validation folds.
    Returns:
      The average testing accuracy of the classifier
      over each fold of cross-validation.
    """
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
#def eval_all_combinations(feature_fns):
    """
    Enumerate all possible classifier settings and compute the
    cross validation accuracy for each setting. We will use this
    to determine which setting has the best accuracy.
    For each setting, construct a LogisticRegression classifier
    and compute its cross-validation accuracy for that setting.
    In addition to looping over possible assignments to
    keep_internal_punct and min_freqs, we will enumerate all
    possible combinations of feature functions. So, if
    feature_fns = [token_features, token_pair_features, lexicon_features],
    then we will consider all 7 combinations of features (see Log.txt
    for more examples).
    Params:
      docs..........The list of original training documents.
      labels........The true labels for each training document (0 or 1)
      punct_vals....List of possible assignments to
                    keep_internal_punct (e.g., [True, False])
      feature_fns...List of possible feature functions to use
      min_freqs.....List of possible min_freq values to use
                    (e.g., [2,5,10])
    Returns:
      A list of dicts, one per combination. Each dict has
      four keys:
      'punct': True or False, the setting of keep_internal_punct
      'features': The list of functions used to compute features.
      'min_freq': The setting of the min_freq parameter.
      'accuracy': The average cross_validation accuracy for this setting, using 5 folds.
      This list should be SORTED in descending order of accuracy.
      This function will take a bit longer to run (~20s for me).
    """
    results = []
    tokens_list = []
    
    # Extract all possible feature combinations
    comb = []
    
    comb = list(chain(*map(lambda x: combinations(feature_fns, x), range(1, len(feature_fns)+1))))
    #for r in range(len(feature_fns)):
   #     comb.append(set(itertools.combinations(feature_fns, r+1)))    
    for punct in punct_vals:

        tokens_list = []
        for doc in docs:

            tokens_list.append(tokenize(doc,punct))
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
    """
    Plot all accuracies from the result of eval_all_combinations
    in ascending order of accuracy.
    Save to "accuracies.png".
    """   
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
    """
    To determine how important each model setting is to overall accuracy,
    we'll compute the mean accuracy of all combinations with a particular
    setting. For example, compute the mean accuracy of all runs with
    min_freq=2.
    Params:
      results...The output of eval_all_combinations
    Returns:
      A list of (accuracy, setting) tuples, SORTED in
      descending order of accuracy.
    """
    #{'punct':punct,'features':features,'min_freq':min_freq,'accuracy':accuracy}
    
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
    """
    Using the best setting from eval_all_combinations,
    re-vectorize all the training data and fit a
    LogisticRegression classifier to all training data.
    (i.e., no cross-validation done here)
    Params:
      docs..........List of training document strings.
      labels........The true labels for each training document (0 or 1)
      best_result...Element of eval_all_combinations
                    with highest accuracy
    Returns:
      clf.....A LogisticRegression classifier fit to all
            training data.
      vocab...The dict from feature name to column index.
    """
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
    """
    Find the n features with the highest coefficients in
    this classifier for this label.
    See the .coef_ attribute of LogisticRegression.
    Params:
      clf.....LogisticRegression classifier
      label...1 or 0; if 1, return the top coefficients
              for the positive class; else for negative.
      n.......The number of coefficients to return.
      vocab...Dict from feature name to column index.
    Returns:
      List of (feature_name, coefficient) tuples, SORTED
      in descending order of the coefficient for the
      given class label.
    """
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
    """
    Using the vocabulary fit to the training data, read
    and vectorize the testing data. Note that vocab should
    be passed to the vectorize function to ensure the feature
    mapping is consistent from training to testing.
    Note: use read_data function defined above to read the
    test data.
    Params:
      best_result...Element of eval_all_combinations
                    with highest accuracy
      vocab.........dict from feature name to column index,
                    built from the training data.
    Returns:
      test_docs.....List of strings, one per testing document,
                    containing the raw.
      test_labels...List of ints, one per testing document,
                    1 for positive, 0 for negative.
      X_test........A csr_matrix representing the features
                    in the test data. Each row is a document,
                    each column is a feature.
                   
    """
    test_docs, test_labels = read_data(os.path.join('data', 'test'))
    punct = best_result['punct']
    feature_fns = best_result['features']
    min_freq = best_result['min_freq']
    tokens_list = []
    
    for doc in test_docs:
        tokens_list.append(tokenize(doc, punct)) 
    X_test,_ = vectorize(tokens_list, feature_fns, min_freq, vocab)
    
    return test_docs, test_labels, X_test
    
def print_top_misclassified(test_docs, test_labels, X_test, clf, n):
    """
    Print the n testing documents that are misclassified by the
    largest margin. By using the .predict_proba function of
    LogisticRegression <https://goo.gl/4WXbYA>, we can get the
    predicted probabilities of each class for each instance.
    We will first identify all incorrectly classified documents,
    then sort them in descending order of the predicted probability
    for the incorrect class.
    E.g., if document i is misclassified as positive, we will
    consider the probability of the positive class when sorting.
    Params:
      test_docs.....List of strings, one per test document
      test_labels...Array of true testing labels
      X_test........csr_matrix for test data
      clf...........LogisticRegression classifier fit on all training
                    data.
      n.............The number of documents to print.
    Returns:
      Nothing; see Log.txt for example printed output.
    """
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
        print(test_docs[index])
    
    
    
def main():
    """
    Put it all together.
    ALREADY DONE.
    """
    
    feature_fns = [token_features, token_pair_features, lexicon_features]
    # Download and read data.
    download_data()
    docs, labels = read_data(os.path.join('data', 'train'))
    # Evaluate accuracy of many combinations
    # of tokenization/featurization.
    results = eval_all_combinations(docs, labels,
                                    [True, False],
                                    feature_fns,
                                    [2,5,10])
    
    # Print information about these results.
    best_result = results[0]
    worst_result = results[-1]
    print('best cross-validation result:\n%s' % str(best_result))
    print('worst cross-validation result:\n%s' % str(worst_result))
    plot_sorted_accuracies(results)
    print('\nMean Accuracies per Setting:')
    print('\n'.join(['%s: %.5f' % (s,v) for v,s in mean_accuracy_per_setting(results)]))

    # Fit best classifier.
    clf, vocab = fit_best_classifier(docs, labels, results[0])

    # Print top coefficients per class.
    print('\nTOP COEFFICIENTS PER CLASS:')
    print('negative words:')
    print('\n'.join(['%s: %.5f' % (t,v) for t,v in top_coefs(clf, 0, 5, vocab)]))
    print('\npositive words:')
    print('\n'.join(['%s: %.5f' % (t,v) for t,v in top_coefs(clf, 1, 5, vocab)]))

    # Parse test data
    test_docs, test_labels, X_test = parse_test_data(best_result, vocab)

    # Evaluate on test set.
    predictions = clf.predict(X_test)
    print('testing accuracy=%f' %
          accuracy_score(test_labels, predictions))

    print('\nTOP MISCLASSIFIED TEST DOCUMENTS:')
    print_top_misclassified(test_docs, test_labels, X_test, clf, 5)


if __name__ == '__main__':
    main()