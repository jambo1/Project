import os
import numpy as np
import scipy as sp
from sklearn.svm import SVC
import pickle
from sklearn.externals import joblib
import ProjectV2.feature_extract as feature_extract
from ProjectV2.comment import Comment
import ProjectV2.topic_build as topic
from time import time


t1=time()
#Read in the saved sarcastic and negative comments numpy arrays
sarcComments = np.load("SarcFiles/25kFiles/sarccoms.npy")
negComments = np.load("SarcFiles/25kFiles/negcoms.npy")

#Load the DictVectorizer and classifier
file1 = "SarcFiles/25kFiles/vectordict.p"
file2 = "SarcFiles/25kFiles/classif_all.p"

vec = joblib.load(file1)
classifier = joblib.load(file2)

topic_mod = topic.topic(model=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'SarcFiles/25kFiles/topics.tp'),\
                        dicttp=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'SarcFiles/25kFiles/topics_dict.tp'))

def getSarcasmScore(sentence):
    sentence = Comment(sentence)
    feature = feature_extract.getallfeatureset(sentence, topic_mod)
    feature_vec = vec.transform(feature)
    score = classifier.decision_function(feature_vec)[0]
    percentage = int(round(2.0 * (1.0 / (1.0 + np.exp(-score)) - 0.5) * 100.0))

    return percentage

sentence1 = "oh i didn't know that"
sentence2 = "hello"
sentence3 = "AMAZING work trump!!!!!! stealing people's kids, how smart"
sentence4 = "You're a genius, how has no one ever thought of this before I wonder"
print(getSarcasmScore(sentence1),"% sarcastic")
print(getSarcasmScore(sentence2),"% sarcastic")
print(getSarcasmScore(sentence3))
print(getSarcasmScore(sentence4))

print("Time taken =", time()-t1)

#def getStats():
