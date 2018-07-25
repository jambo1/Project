from ProjectV2.comment import Comment
import ProjectV2.get_texts as GetTexts
import ProjectV2.topic_build as Topical
import numpy as np
import ProjectV2.feature_extract as feature_extract
import scipy as sp
import pickle
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction import DictVectorizer


#Read the file
print("Reading file")

##Uncomment for the bigger files
#sarcComments = GetTexts.read_sarc_file('BigSarc.csv')
#negComments = GetTexts.read_neg_file('BigNeg.csv')

#These are the smaller files
sarcComments = GetTexts.read_sarc_file("Sarc Set.csv")
negComments = GetTexts.read_neg_file("Non Sarc Set.csv")

##Uncomment the following when the desired data is already in the system
#sarcComments = np.load("sarccoms.npy")
#negComments = np.load("negcoms.npy")

print("Done reading file")

#Pass the comments to the topic to fit the data
print("Topic time")
topic_mod = Topical.topic(nbtopic=200,alpha='symmetric')
topic_mod.fit(np.concatenate((sarcComments,negComments)))

#Print 20 of the topics
print("Topics finished properly")
for i in range (0,20):
    print(topic_mod.get_topic(i))

print("The feature is beginning")

cls_set = ['Non-Sarcastic', 'Sarcastic']
featuresets = []

#Deal with sarcastic comments
index = 0
for com in sarcComments:
    # com = com.text
    featuresets.append((feature_extract.getallfeatureset(com, topic_mod), cls_set[1]))
    index += 1

#Deal with non sarcastic comments
index = 0
for com in negComments:
    #com = com.text
    featuresets.append((feature_extract.getallfeatureset(com, topic_mod), cls_set[0]))
    index += 1

#Check the length of featuresets
print(len(featuresets))

print("Turning feature to np array")
featuresets = np.array(featuresets)
print(featuresets)

#Check shape of np array
print(featuresets.shape)

print("Setting targets")
labels = (featuresets[0::,1] == 'Sarcastic').astype(int)

#Turn NP array into a DictVectorizer, then save it
print ("Making vect dict")
vec = DictVectorizer()
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
train_targets = labels[size:]
testvec = featurevec[:size,0::]
test_targets = labels[:size]

#The following works with above 60% accuracy
print("Set the training data")
#Artificial weights
pos_p=(train_targets==1)
neg_p=(train_targets==0)
ratio = np.sum(neg_p.astype(float))/np.sum(pos_p.astype(float))
new_trainvec = trainvec
new_train_targets=train_targets
for j in range(int(ratio-1.0)):
    new_trainvec=sp.sparse.vstack([new_trainvec,trainvec[pos_p,0::]])
    new_train_targets=np.concatenate((new_train_targets, train_targets[pos_p]))

print("Fit the classifier")
classifier = SVC(C=0.1,kernel='linear')
classifier.fit(new_trainvec,new_train_targets)

print("Saving the classifier")
#Saving the classifier
file_Name = "classif_all.p"
fileObject = open(file_Name,'wb')
pickle.dump(classifier, fileObject)
fileObject.close()

print ('Validating')

output = classifier.predict(testvec)
clfreport = classification_report(test_targets, output, target_names=cls_set)
print (clfreport)
print(accuracy_score(test_targets, output)*100)