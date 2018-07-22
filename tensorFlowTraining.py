import tensorflow as tf
from tensorflow.python.data import Dataset
import ProjectV2.get_texts as GetTexts
import ProjectV2.topic_build as Topical
import numpy as np
import ProjectV2.feature_extract as feature_extract
import scipy as sp
from sklearn.feature_extraction import DictVectorizer
import pickle
from sklearn.utils import shuffle
import ProjectV2.tensorMethods as tensorMethods

tf.logging.set_verbosity(tf.logging.ERROR)

#Read the file
print("Reading file")

##Uncomment for the bigger files
#sarcComments = GetTexts.read_sarc_file('BigSarc.csv')
#negComments = GetTexts.read_neg_file('BigNeg.csv')

#These are the smaller files
sarcComments = GetTexts.read_sarc_file("Sarc Set.csv")
negComments = GetTexts.read_sarc_file("Non Sarc Set.csv")

##Uncomment the following when the desired data is already in the system
#sarcComments = np.load("sarccoms.npy")
#negComments = np.load("negcoms.npy")

print("Done reading file")

#Pass the comments to the topic to fit the data
print("Topic time")
topic_mod = Topical.topic(nbtopic=200,alpha='symmetric')
topic_mod.fit(np.concatenate((sarcComments,negComments)))

print("The feature is beginning")

cls_set = ['Non-Sarcastic', 'Sarcastic']
featuresets = []

#Deal with sarcastic comments
index = 0
for com in sarcComments:
    featuresets.append((feature_extract.getallfeatureset(com, topic_mod), cls_set[1]))
    index += 1

#Deal with non sarcastic comments
index = 0
for com in negComments:
    com = com.text
    featuresets.append((feature_extract.getallfeatureset(com, topic_mod), cls_set[0]))
    index += 1

#Check the length of featuresets
print(len(featuresets))

print("Turning feature to np array")
featuresets = np.array(featuresets)
print(featuresets)

print("Setting targets")
labels = (featuresets[0::, 1] == 'Sarcastic').astype(int)

#Turn NP array into a DictVectorizer, then save it
print ("Making vect dict")
vec = DictVectorizer()

#If using a different dataset should this be changed or remain the same or be added to?
try:
    featurevec = pickle.load("vectordict.p")
except:
    featurevec = vec.fit_transform(featuresets[0::,0])
    print("Saving vect dict")
    file_Name = "vectordict.p"
    fileObject = open(file_Name, 'wb')
    pickle.dump(vec, fileObject)
    fileObject.close()

print("Splitting the featuresets into training and testing")
order=shuffle(range(len(featuresets)))
labels=labels[order]
featurevec=featurevec[order,0::]

#Spliting
size = int(len(featuresets) * .3) # 30% is used for the test set

print("Setting training and test targets and vectors")
trainvec = featurevec[size:,0::]
train_labels = labels[size:]
testvec = featurevec[:size,0::]
test_labels = labels[:size]

print("Set the training data")
#Artificial weights
pos_p=(train_labels==1)
neg_p=(train_labels==0)
ratio = np.sum(neg_p.astype(float))/np.sum(pos_p.astype(float))
new_trainvec=trainvec
new_train_labels=train_labels
new_testvec=testvec
new_test_labels=test_labels

for j in range(int(ratio-1.0)):
    new_trainvec=sp.sparse.vstack([new_trainvec,trainvec[pos_p,0::]])
    new_train_labels=np.concatenate((new_train_labels,train_labels[pos_p]))
    new_testvec = sp.sparse.vstack([new_testvec, testvec[pos_p, 0::]])
    new_test_labels = np.concatenate((new_test_labels, test_labels[pos_p]))

print("Regressing")
linear_regressor = tensorMethods.train_model(
    learning_rate=0.00003,
    steps=500,
    batch_size=5,
    training_examples=new_trainvec,
    training_targets=new_train_labels,
    validation_examples=testvec,
    validation_targets=test_labels)

