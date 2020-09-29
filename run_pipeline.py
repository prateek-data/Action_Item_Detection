#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 11:10:52 2020

@author: user
"""

import pandas as pd

from data_preprocessing import process_enron_sample
from nlp_rule_match import spacy_check_email_list
from ml_classifier import data_loader, get_features, data_split, classify_logistic, classify_random_forest
from dl_models import data_processing, load_embeddings, train_lstm, save__model, load__model, evaluate_model



inputCsv = "Datasets/emails.csv"
actionCsv = "Datasets/actions.csv"

#data preparation
##work on a sample of 5k emails from enron dataset due to memory constraints
sentenceList = process_enron_sample(inputCsv, sampleSize = 3000)

#run nlp matcher on sentenceList to classify actionable and non actionable sentences
tempList = sentenceList[0:3000]
nlpResultList =  spacy_check_email_list(tempList)


#Using ML Models
##Load data
actionDf = pd.read_csv(actionCsv, names = ['Text'])
actionList = actionDf['Text'].to_list()

allText = []
for text in actionList:
    rowDict = {}
    rowDict['Text'] = text
    rowDict['Ground_Truth'] = True
    allText.append(rowDict)
 
counter = 0    
for item in nlpResultList:
    if item[1] == False and counter < 1250:
        rowDict = {}
        rowDict['Text'] = item[0]
        rowDict['Ground_Truth'] = False
        allText.append(rowDict)
        counter+=1
        
#prepare training and testing data
textList, gtList = data_loader(allText)
featureArray, gtArray = get_features(textList, gtList)
trainX, testX, trainY, testY = data_split(featureArray, gtArray)

#train classifiers and print accuracy on test set
##logistic regression model
logClf, logMetrics = classify_logistic(trainX, testX, trainY, testY)
print("Accuracy from logistic classifier : ", logMetrics)

##random forest model
rfClf, rfMetrics = classify_random_forest(trainX, testX, trainY, testY)
print("Accuracy from random forest classifier : ", rfMetrics)


#######################################################################################################
print("###############################################################################################")
#######################################################################################################

#Using Deep Learning Models
##prepare training and testing data
textList, gtList = data_loader(allText)
tokenizer, maxLen, vocabSize, featureArray, gtArray = data_processing(textList, gtList)
trainX, testX, trainY, testY = data_split(featureArray, gtArray)


##LSTM model
####Uncomment the code block below to train LSTM model
''' 
embeddingMatrix = load_embeddings(tokenizer, vocabSize)
lstmModel = train_lstm(trainX, trainY, maxLen, vocabSize, embeddingMatrix)
save__model(lstmModel, "Saved_Models/Lstm_Model")
'''

#load LSTM model
lstmModel = load__model("Saved_Models/Lstm_Model")
#evaluate LSTM model
lstmMetrics = evaluate_model(lstmModel, testX, testY)
print(lstmMetrics)



##CNN Model
####Uncomment the code block below to train CNN model
'''
embeddingMatrix = load_embeddings(tokenizer, vocabSize)
cnnModel = train_cnn(trainX, trainY, maxLen, vocabSize, embeddingMatrix)
save__model(cnnModel, "Saved_Models/Cnn_Model")
'''

#Load CNN model
cnnModel = load__model("Saved_Models/Cnn_Model")
#evaluate CNN model
cnnMetrics = evaluate_model(cnnModel, testX, testY)
print(cnnMetrics)


#######################################################################################################
print("###############################################################################################")
#######################################################################################################

        
    

    





