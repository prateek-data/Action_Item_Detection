#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 11:09:10 2020

@author: user
"""

import spacy
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


nlp = spacy.load('en_core_web_md') 


def data_loader(allText):
    textList = []
    gtList = []
    for item in allText:
        textList.append(item['Text'])
        gtList.append(item['Ground_Truth'])
    return textList, gtList

def get_features(textList, gtList):    
    tagList = []
    posList = []     
    for text in textList:
        posString = ""
        tagString = ""
        doc = nlp(text)
        
        for token in doc:
            if token == doc[-1]:
                posString += (token.pos_)
                tagString += (token.tag_)
            else:
                posString += (token.pos_ + " ")
                tagString += (token.tag_ + " ")

        posList.append(posString)    
        tagList.append(tagString)
        
    #prepare TF-IDF vectors
    vectorizer1 = TfidfVectorizer(ngram_range = (2,4))
    vectorizer2 = TfidfVectorizer(ngram_range = (2,2))
    #posArray = vectorizer.fit_transform(posList).toarray()
    tagArray = vectorizer1.fit_transform(tagList).toarray()
    textArray = vectorizer2.fit_transform(textList).toarray()
    
    
    #prepare features and labels
    
    featureArray = np.column_stack((tagArray, textArray))
    #featureArray = tagArray
    gtList = [1 if item == True else 0 for item in gtList ]
    gtArray = np.array(gtList)
    
    return (featureArray, gtArray)
    
   
def data_split(featureArray, gtArray):
     trainX, testX, trainY, testY =  train_test_split(featureArray, gtArray, test_size = 0.25, random_state= 42)
     return (trainX, testX, trainY, testY)
 
 
def classify_logistic(trainX, testX, trainY, testY):
    clf = LogisticRegression().fit(trainX, trainY)
    pred = clf.predict(testX)
    
    acc = accuracy_score(testY, pred)
    print(acc)
    return (clf, acc)
    
    
def classify_random_forest(trainX, testX, trainY, testY):
    clf = RandomForestClassifier(max_depth=2).fit(trainX, trainY)
    pred = clf.predict(testX)
    
    acc = accuracy_score(testY, pred)
    print(acc)
    return (clf, acc)
    

def save_model(clf, name):
    with open('Saved_Models/' + name + '.pkl', 'wb') as f:
        pickle.dump(clf, f)
    
   
    
    
        
    
text = 'She is available for inter views about her familys flight from Saigon and her American success story.'
text  = "Hello how are you feeling today?"
doc = nlp(text)
tagString = ""
for token in doc:
    if token == doc[-1]:
        tagString += (token.tag_)
    else:
        tagString += (token.tag_ + " ")




text2tagArray = vectorizer1.transform([tagString]).toarray()
text2textArray = vectorizer2.transform([text]).toarray()
    
text2featureArray =  np.column_stack((text2tagArray, text2textArray))   
#text2featureArray = text2tagArray 
textPred = clf.predict(text2featureArray)
print(textPred)
        
        
        
        
        
        
        
        
    



