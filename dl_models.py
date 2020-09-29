#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 20:30:13 2020

@author: user
"""
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, load_model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.layers import Conv1D, MaxPooling1D, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support



allTextdf =  pd.read_csv("allText.csv")
allText = list(allTextdf.T.to_dict().values())
for d in allText:
    del d['Unnamed: 0']
    

def data_loader(allText):
    textList = []
    gtList = []
    for item in allText:
        textList.append(item['Text'])
        gtList.append(item['Ground_Truth'])
    return textList, gtList


def data_split(featureArray, gtArray):
     trainX, testX, trainY, testY =  train_test_split(featureArray, gtArray, test_size = 0.25, random_state= 42)
     return (trainX, testX, trainY, testY)

def data_processing(textList, gtList):
    #prepare data + load glove embeddings for lstm model
    ##find max sentence length in corpus
    maxLen = 0
    for text in textList:
        if len(text.split()) > maxLen:
            maxLen = len(text.split())
            
    
    #tokenize corpus and get size of vocabulary
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(textList)
    sequences = tokenizer.texts_to_sequences(textList)
    vocabSize = len(tokenizer.word_index) + 1
    
    featureData = pad_sequences(sequences, padding = 'post', maxlen = maxLen)
    gtList = [1 if item == True else 0 for item in gtList ]
    gtArray = np.array(gtList)
    
    return tokenizer, maxLen, vocabSize, featureData, gtArray


def load_embeddings(tokenizer, vocabSize):
    #importing glove embeddings and loading them into a dictionary 
    embeddingsIndex = dict()
    f = open('Datasets/glove.6B/glove.6B.100d.txt')
    for line in f:
    	values = line.split()
    	word = values[0]
    	coefs = np.asarray(values[1:], dtype='float32')
    	embeddingsIndex[word] = coefs
    f.close()
    print('Found %s word vectors.' % len(embeddingsIndex))
    
    
    # create a glove weight mapping for words in corpus
    embeddingMatrix = np.zeros((vocabSize, 100))
    for word, i in tokenizer.word_index.items():
    	embeddingVector = embeddingsIndex.get(word)
    	if embeddingVector is not None:
    		embeddingMatrix[i] = embeddingVector 
            
    
    return embeddingMatrix
            
    
def train_lstm(trainX, trainY, maxLen, vocabSize, embeddingMatrix):
    #setting model hyperparameters
    embeddingDim = 100
    hiddenDim = 60
    epochs = 30
    batchSize = 64    

    #create model architecture
    
    model = Sequential()
    model.add(Embedding(input_dim=vocabSize, output_dim=embeddingDim, input_length=maxLen, weights=[embeddingMatrix], trainable=False))
    model.add(LSTM(hiddenDim, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    #train model
    history = model.fit(trainX, trainY, epochs=epochs, batch_size=batchSize, validation_split=0.1)
    #history = model.fit(trainX, trainY, epochs=epochs, batch_size=batchSize, validation_split=0.1, callbacks = ModelCheckpoint(filepath="Saved_Models/Lstm_Model_Weights.hdf5" ,monitor='val_accuracy', save_best_only=True))
    #history = model.fit(trainX, trainY, epochs=epochs, batch_size=batchSize, validation_split=0.1, callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])
    return model


def train_cnn(trainX, trainY, maxLen, vocabSize, embeddingMatrix):
    #setting model hyperparameters
    embeddingDim = 100
    filters = 64
    kernelSize = 5
    epochs = 30
    batchSize = 64  
    
    #create model architecture
    model = Sequential()
    model.add(Embedding(input_dim=vocabSize, output_dim=embeddingDim, input_length=maxLen, weights=[embeddingMatrix], trainable=False))
    model.add(Conv1D(filters, kernelSize, activation='relu'))
    model.add(Dropout(0.2))
    model.add(MaxPooling1D(pool_size=5))
    model.add(Flatten())
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    
    #train_model
    history = model.fit(trainX, trainY, epochs=epochs, batch_size=batchSize, validation_split=0.1)
    return model
    
    

def save_dl_model(model, loc):
    model.save(loc)
    
def load_dl_model(loc):
    model = load_model(loc)
    return model

    

def evaluate_model(model, testX, testY):
    #evaluate model performance metrics on test set
    acc = model.evaluate(testX, testY)
    print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(acc[0],acc[1]))
    
    pred = model.predict_classes(testX, verbose=0)
    precision, recall, f1score, support = precision_recall_fscore_support(testY, pred, average='binary')
    
    metrics = {}
    metrics['Accuracy'] = acc[1]
    metrics['Precision'] = precision
    metrics['Recall'] = recall
    metrics['F1Score'] = f1score
    return metrics
    

    
def run_dl_inference(model, sentence):
    #run inference on deep learning model
    allTextdf =  pd.read_csv("allText.csv")
    allText = list(allTextdf.T.to_dict().values())
    for d in allText:
        del d['Unnamed: 0']
        
    textList = []
    gtList = []
    for item in allText:
        textList.append(item['Text'])
        gtList.append(item['Ground_Truth'])
        
    tokenizer, maxLen, vocabSize, featureArray, gtArray = data_processing(textList, gtList)
    inputSequence = tokenizer.texts_to_sequences([sentence])
    inputSequence = pad_sequences(inputSequence, padding = 'post', maxlen = maxLen)
    
    pred = model.predict_classes(inputSequence)
    
    if pred[0][0] == 1:
        return "Actionable"
    else:
        return "Non Actionable"
    
    
    
        
    
    