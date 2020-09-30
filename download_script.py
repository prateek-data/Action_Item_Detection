#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 14:35:21 2020

@author: user
"""

import os
import urllib
import progressbar
import nltk
import spacy


def show_progress(block_num, block_size, total_size):
    #function to show download progress
    global pbar
    if pbar is None:
        pbar = progressbar.ProgressBar(maxval=total_size)
        pbar.start()

    downloaded = block_num * block_size
    if downloaded < total_size:
        pbar.update(downloaded)
    else:
        pbar.finish()
        pbar = None



#download spacy md model
spacy.cli.download("en_core_web_md")

#download nltk dependencies
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


#download Enron dataset from Kaggle 
pbar = None
url = "https://www.kaggle.com/wcukierski/enron-email-dataset"
fileDest = "temp_data"

urllib.request.urlretrieve(url, os.path.join(fileDest, 'archive.zip'), show_progress)


#download Glove vectors
pbar = None
url = "http://nlp.stanford.edu/data/glove.6B.zip"
fileDest = "temp_data"

urllib.request.urlretrieve(url, os.path.join(fileDest, 'glove.6B.zip'), show_progress)