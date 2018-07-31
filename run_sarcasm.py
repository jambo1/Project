import os
import numpy as np
import scipy as sp
from sklearn.svm import SVC
import pickle
import ProjectV2.feature_extract as feature_extract
from ProjectV2.comment import Comment
import ProjectV2.topic_build as topic



#Read in the saved sarcastic and negative comments numpy arrays
sarcComments = np.load("sarccoms.npy")
negComments = np.load("negcoms.npy")

#Load the DictVectorizer and classifier
file1 = "vectordict.p"
file2 = "classif_all.p"
vecObject = open(file1, 'rb')
classifObject = open(file2, 'rb')
vec = pickle.load(vecObject)
classifier = pickle.load(classifObject)
vecObject.close()
classifObject.close()

topic_mod = topic.topic(model=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'topics.tp'),\
                        dicttp=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'topics_dict.tp'))

def getSarcasmScore(sentence):
    sentence = Comment(sentence)
    feature = feature_extract.getallfeatureset(sentence, topic_mod)
    feature_vec = vec.transform(feature)
    score = classifier.decision_function(feature_vec)[0]
    percentage = int(round(2.0 * (1.0 / (1.0 + np.exp(-score)) - 0.5) * 100.0))

    return percentage

sentence1 = "oh i didn't know that"
sentence2 = "the world is going to end tomorrow"
sentence3 = "AMAZING work trump!!!!!! stealing people's kids, how smart"
sentence4 = "You're a genius, how has no one ever thought of this before I wonder"
print(getSarcasmScore(sentence1),"% sarcastic")
print(getSarcasmScore(sentence2),"% sarcastic")
print(getSarcasmScore(sentence3))
print(getSarcasmScore(sentence4))