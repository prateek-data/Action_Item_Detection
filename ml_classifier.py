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
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support


nlp = spacy.load('en_core_web_md') 


def data_loader(allText):
    #load data
    textList = []
    gtList = []
    for item in allText:
        textList.append(item['Text'])
        gtList.append(item['Ground_Truth'])
    return textList, gtList

def get_features(textList, gtList):    
    #prepare features
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
     trainX, testX, trainY, testY =  train_test_split(featureArray, gtArray, test_size = 0.2, random_state= 42)
     return (trainX, testX, trainY, testY)
 
 
def classify_logistic(trainX, trainY, featureArray, gtArray):
    #train logistic regression classifier and test using kfold cross validation
    clf = LogisticRegression().fit(trainX, trainY)    
    pred = cross_val_predict(clf, featureArray, gtArray, cv = 5)
    
    acc = accuracy_score(gtArray, pred)
    print(acc)
    precision, recall, f1score, support = precision_recall_fscore_support(gtArray, pred, average='binary')
    
    metrics = {}
    metrics['Accuracy'] = acc
    metrics['Precision'] = precision
    metrics['Recall'] = recall
    metrics['F1Score'] = f1score
    
    return (clf, metrics)
    
    
def classify_random_forest(trainX, trainY, featureArray, gtArray):
    #train and test random forest classifier
    clf = RandomForestClassifier(max_depth=30).fit(trainX, trainY)
    pred = cross_val_predict(clf, featureArray, gtArray, cv = 5)
        
    acc = accuracy_score(gtArray, pred)
    print(acc)
    
    precision, recall, f1score, support = precision_recall_fscore_support(gtArray, pred, average='binary')
    
    metrics = {}
    metrics['Accuracy'] = acc
    metrics['Precision'] = precision
    metrics['Recall'] = recall
    metrics['F1Score'] = f1score
    
    return (clf, metrics)
    

def save_ml_model(clf, name):
    #save serialized version of ml model
    with open('Saved_Models/' + name + '.pkl', 'wb') as f:
        pickle.dump(clf, f)
        
        
def load_ml_model(name):
    #load serialized version of ml model
    with open('Saved_Models/' + name + '.pkl', 'rb') as f:
        clf = pickle.load(f) 
    return clf
    
    
def run_ml_inference(clf, sentence):
    #run inference on ml classifier
    allTextdf =  pd.read_csv("Datasets/allText.csv")
    allText = list(allTextdf.T.to_dict().values())
    for d in allText:
        del d['Unnamed: 0']
        
    textList, gtList = data_loader(allText)
    
    tagList = []   
    for text in textList:
        tagString = ""
        doc = nlp(text)
        
        for token in doc:
            if token == doc[-1]:
                tagString += (token.tag_)
            else:
                tagString += (token.tag_ + " ")
    
        tagList.append(tagString)
        
    #prepare TF-IDF vectors
    vectorizer1 = TfidfVectorizer(ngram_range = (2,4))
    vectorizer2 = TfidfVectorizer(ngram_range = (2,2))
    #posArray = vectorizer.fit_transform(posList).toarray()
    vectorizer1.fit_transform(tagList).toarray()
    vectorizer2.fit_transform(textList).toarray()
    
    
    doc = nlp(sentence)
    tagString = ""
    for token in doc:
        if token == doc[-1]:
            tagString += (token.tag_)
        else:
            tagString += (token.tag_ + " ")
    text2tagArray = vectorizer1.transform([tagString]).toarray()
    text2textArray = vectorizer2.transform([sentence]).toarray()
    
    text2featureArray =  np.column_stack((text2tagArray, text2textArray))   
    pred = clf.predict(text2featureArray)
    
    if pred[0] == 1:
        return "Actionable"
    else:
        return "Non Actionable"
        
    

        
        
        
        
        
        
        
        
    



