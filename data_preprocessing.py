#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 11:16:04 2020

@author: user
"""

#libraries imported
import email
import re
import pandas as pd
import pickle
import nltk
import random


ipCsv = "Datasets/emails.csv"
actionCsv = "Datasets/actions.csv"


def process_enron(ipCsv):
    #load dataframe and extract email body
    enronDf = pd.read_csv(ipCsv)    
    enronDf['Email Body'] = enronDf['message'].apply(lambda x: email.message_from_string(x).get_payload())
    emailBodyList = enronDf['Email Body'].tolist()
    
    #data cleaning
    ##removing original messages and forwarded messages from emails
    updatedEmailList = []
    for mail in emailBodyList:
        if ("Forwarded by" in mail) or ("Original Message" in mail): 
            try:
                mail = mail.split("Forwarded by")[0]
            except:
                pass
            
            try:
                mail = mail.split("Original Message")[0]
            except:
                pass  
            
        updatedEmailList.append(mail)
          
    ##removing special characters and cleaning text
    for id, mail in enumerate(updatedEmailList):
        mail = re.sub("[\n\t-]", " " , mail)
        mail = re.sub("[^0-9a-zA-Z (),?!.]", "", mail)
        updatedEmailList[id] = " ".join(mail.split())
        
        
    #tokenizing email texts into sentences    
    sentList = []
    for idx, mail in enumerate(updatedEmailList):
        updatedEmailList[idx] = nltk.sent_tokenize(mail)
               
    sentList = [item for sublist in updatedEmailList for item in sublist]     
    return sentList
    

def process_enron_sample(ipCsv, sampleSize):
    #process a random sample of enron emails due to memeory constraints
    ##load dataframe and extract email body
    enronDf = pd.read_csv(ipCsv)  
    enronDf = enronDf.sample(n= sampleSize, random_state=1)
    enronDf['Email Body'] = enronDf['message'].apply(lambda x: email.message_from_string(x).get_payload())
    emailBodyList = enronDf['Email Body'].tolist()
    
    #data cleaning
    ##removing original messages and forwarded messages from emails
    updatedEmailList = []
    for mail in emailBodyList:
        if ("Forwarded by" in mail) or ("Original Message" in mail): 
            try:
                mail = mail.split("Forwarded by")[0]
            except:
                pass
            
            try:
                mail = mail.split("Original Message")[0]
            except:
                pass  
            
        updatedEmailList.append(mail)
          
    ##removing special characters and cleaning text
    for id, mail in enumerate(updatedEmailList):
        mail = re.sub("[\n\t-]", " " , mail)
        mail = re.sub("[^0-9a-zA-Z (),?!.]", "", mail)
        updatedEmailList[id] = " ".join(mail.split())
        
        
    #tokenizing email texts into sentences where sentences contain less than 35 words    
    sentList = []
    for idx, mail in enumerate(updatedEmailList):
        updatedEmailList[idx] = nltk.sent_tokenize(mail)
              
    sentList = [item for sublist in updatedEmailList for item in sublist if (3 < len(item.split()) <= 35) ]        
    return sentList
    


def save_sentences(sentList): 
    #serialize sentence list 
    with open('Datasets/sentencelist.pkl', 'wb') as f:
        pickle.dump(sentList, f)
       
       
       
def sample_sentences(sentList, sampleSize):
    #randomly sample 10k sentences from sentence list
    sampleList = random.sample(sentList, k= sampleSize)
    return sampleList


    
