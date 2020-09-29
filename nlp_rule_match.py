#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 12:48:15 2020

@author: user
"""

import nltk
from nltk import RegexpParser
from nltk.tree import Tree
import re
import spacy
nlp = spacy.load('en_core_web_md') 

firstWordQues = ['when', 'why', 'will', 'wont', 'where', 'is', 'who', 'do' , 'can', 'could', 'would']

def chunk_matcher(posTaggedSent):
    #define candidate chunks to match in sentences
    candidateChunks = r"""
                        Case1: {<VB><DT><JJ>*<NN.?>}
                        Case2: {<VB><VBG><DT><JJ>*<NN.?>}
                        Case3: {<VB><PRP><DT><JJ>*<NN.?>}
                        Case4: {<MD><PRP><VB>}
                      """  
              
    chunkparser = RegexpParser(candidateChunks)
    return chunkparser.parse(posTaggedSent)  


def nltk_matcher(posTaggedSent): 
    #conditions for action item classification
    if posTaggedSent[0][1] == "VB" and posTaggedSent[-1][0] != "?":
        return True
    
    elif posTaggedSent[0][0].lower() in firstWordQues:
        return False
    
    else:
        chunks = chunk_matcher(posTaggedSent)
        for element in chunks:
            if type(element) == Tree and element.label() in ["Case1", "Case2", "Case3", "Case4"]:
                #print(element.label())
                return True
            
        else:
            return False
        

def spacy_matcher(doc):
    if (doc[0].tag_ == 'VB') and (doc[-1].text != '?'):
        return True
    
    elif doc[0].text.lower() in firstWordQues:
        return False
    
    else:
        tagList = []
        for token in doc:
            tagList.append(token.tag_)
        tagString = " ".join(tagList)
        matchObj = re.search(r"(VB DT (JJ )*NN.?)|(VB PRP DT (JJ )*NN.?)|(UH VB)", tagString)
        if matchObj != None:
            return True
        
        else:
            return False
    
        
def nltk_check_sentence(sentence):
    #checks if individual sentence is actionable or not using nltk 
    tokenList = nltk.word_tokenize(sentence)
    posTaggedSent = nltk.pos_tag(tokenList)    
    result = nltk_matcher(posTaggedSent)
    return result

    
def nltk_check_email_list(sentList):
    #run nlp algorithm on list of enron emails to find actionable and non actionable sentences
    resultList = []
    for sentence in sentList:
        tokenList = nltk.word_tokenize(sentence)
        posTaggedSent = nltk.pos_tag(tokenList)    
        result = nltk_matcher(posTaggedSent) 
        resultList.append((sentence, result))
    return resultList


def spacy_check_sentence(sentence):
    #checks if individual sentence is actionable or not using spacy
    doc = nlp(sentence)
    result = spacy_matcher(doc)
    return result
    
    
def spacy_check_email_list(sentList):
    #run nlp algorithm on list of enron emails to find actionable and non actionable sentences        
    resultList = []
    for sentence in sentList:
        doc = nlp(sentence)
        result = spacy_matcher(doc)
        resultList.append((sentence, result))
    return resultList        
    


