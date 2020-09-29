#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 13:42:12 2020

@author: user
"""

import flask
from flask import request

from nlp_rule_match import spacy_check_sentence
from ml_classifier import load_ml_model, run_ml_inference
from dl_models import load_dl_model, run_dl_inference

app = flask.Flask(__name__)
app.config["DEBUG"] = True
app.config['JSON_SORT_KEYS'] = False


@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/nlp_matcher', methods=['POST'])
def nlp_matcher():
    #api for nlp rule based matcher
    assert request.path == '/nlp_matcher'
    assert request.method == 'POST'
    
    
    data = request.json    
    sentence = data['Text']
    
    result = spacy_check_sentence(sentence)
    
    if result != None:
        resDict = {}
        resDict['Text'] = sentence
        resDict['Is_Actionable'] = result
        return (resDict)
   
    
@app.route('/ml_classifier', methods=['POST'])
def ml_classifier():
    #api for ml classifier 
    assert request.path == '/ml_classifier'
    assert request.method == 'POST'
    
    data = request.json    
    sentence = data['Text']
    
    #load model
    rfClf =  load_ml_model("Random_Forest_Classifier")
    result = run_ml_inference(rfClf, sentence)
    del rfClf
    
    if result != None:
        resDict = {}
        resDict['Text'] = sentence
        resDict['Is_Actionable'] = result
        return (resDict)
 
    
@app.route('/dl_model', methods=['POST'])
def dl_model():
    #api for dl model 
    assert request.path == '/dl_model'
    assert request.method == 'POST'
    
    data = request.json    
    sentence = data['Text']
    
    #load model
    lstmModel =  load_dl_model("Saved_Models/Lstm_Model")
    result = run_dl_inference(lstmModel, sentence)
    del lstmModel
    
    if result != None:
        resDict = {}
        resDict['Text'] = sentence
        resDict['Is_Actionable'] = result
        return (resDict)
    
   

if __name__ == '__main__':
    app.run(debug=True , port= 5000) 
    
    