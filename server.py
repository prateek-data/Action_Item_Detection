#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 13:42:12 2020

@author: user
"""

import flask
from flask import request

from nlp_rule_match import check_sentence

app = flask.Flask(__name__)
app.config["DEBUG"] = True
app.config['JSON_SORT_KEYS'] = False


@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/nlp_matcher', methods=['POST'])
def get_input_request():
    
    assert request.path == '/nlp_matcher'
    assert request.method == 'POST'
    
    data = request.json    
    sentence = data['Text']
    result = check_sentence(sentence)
    
    if result != None:
        resDict = {}
        resDict['Text'] = sentence
        resDict['Is_Actionable'] = result
    
        return (resDict)
    
    
    
if __name__ == '__main__':
    app.run(debug=True , port= 5000) 
    
    