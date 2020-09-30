#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 14:35:21 2020

@author: user
"""
# Code to download datasets and additional dependencies.
# Will take some time to execute based of internet speed.
import os
import nltk
import spacy


#download spacy md model
spacy.cli.download("en_core_web_md")
print("Downloaded Spacy en_core_web_md model.") 


#download nltk dependencies
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
print("Downloaded NLTK dependencies.")

