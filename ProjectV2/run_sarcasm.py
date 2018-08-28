import os
import sys
import numpy as np
import scipy as sp
from sklearn.svm import SVC
import pickle
from sklearn.externals import joblib
import feature_extract as feature_extract
from comment import Comment
import topic_build as topic

'''
This module is intended to be used to process a single line and receive a sarcasm score, the getSarcasmScore method handles this.
To use the commandline interface pass any system arguments when running this module
'''

#Load the DictVectorizer and classifier
file1 = "WebSite/app/SarcFiles/25kFiles/vectordict.p"
file2 = "WebSite/app/SarcFiles/25kFiles/classif_all.p"

vec = joblib.load(file1)
classifier = joblib.load(file2)

topic_mod = topic.topic(model=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'WebSite/app/SarcFiles/25kFiles/topics.tp'),\
                        dicttp=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'WebSite/app/SarcFiles/25kFiles/topics_dict.tp'))

def getSarcasmScore(sentence):
    sentence = Comment(sentence)
    feature = feature_extract.getallfeatureset(sentence, topic_mod)
    feature_vec = vec.transform(feature)
    score = classifier.decision_function(feature_vec)[0]
    percentage = int(round(2.0 * (1.0 / (1.0 + np.exp(-score)) - 0.5) * 100.0))

    return percentage

#The command line interface, activated by including any system arguments
if len(sys.argv)>1:
    user_go = True
    while user_go==True:
        #User inputs a sentence and gets a score
        user_input=input("Enter a sentence to determine its sarcasm score: ")
        print("The percentage of sarcasm in the sentence was: ", getSarcasmScore(user_input))

        #Ask user if they want to continue
        while True:
            carry_on = input("Continue? y/n: ").lower()
            if carry_on=="y":
                break
            elif carry_on=="n":
                user_go=False
                print("Thank you and goodbye!")
                break
            else:
                continue





