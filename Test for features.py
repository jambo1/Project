from ProjectV2.comment import Comment
import ProjectV2.get_texts as GetTexts
import ProjectV2.topic_build as Topical
import numpy as np
import ProjectV2.feature_extract as feature_extract
import scipy as sp
from sklearn.utils import shuffle
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.feature_extraction import DictVectorizer
import pickle


#These are the smaller files
sarcComments = GetTexts.read_sarc_file("BigSarc.csv")
negComments = GetTexts.read_sarc_file("BigNeg.csv")

print(len(sarcComments))

sarcLols =0
negLols = 0

laughing = ["lmao", "lmfao", "pmsl", "rofl", "lol"]

for comment in sarcComments:
    sentence = comment.text
    for word in sentence.split():
        if word in laughing:
            sarcLols += 1

for comment in negComments:
    sentence = comment.text
    for word in sentence.split():
        if word in laughing:
            negLols +=1

print("Sarc ", sarcLols, " Nonsarc ", negLols)
