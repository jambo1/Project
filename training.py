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
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction import DictVectorizer
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB


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

#Check shape of np array
print(featuresets.shape)

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

##How to make this work? Would be beneficial for choosing model and features
# #Visualise the data
# pandasets = pd.DataFrame(featurevec)
# print(pandasets.describe())
# print(pandasets.head(20))
#
# pandasets.plot(kind = 'box', subplots=True, layout=(2,2),sharex=False,sharey=False)
# plt.show()
# #histograms
# pandasets.hist()
# plt.show()
# #Multivariates, scatter plot matrix
# scatter_matrix(pandasets)
# plt.show()


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

# ##From Machine Learning Mastery, does not work but would be good for choosing model
# #Test options and evaluation metric
# #random seed number does not matter
# seed=7
# #accuracy will be the number predicted correctly out of the total predictions as a percentage. The scoring variable will be used in running and building models
# scoring = 'accuracy'
#
# models = []
# models.append(('LR', LogisticRegression()))
# models.append(('LDA', LinearDiscriminantAnalysis()))
# models.append(('KNN', KNeighborsClassifier()))
# models.append(('CART', DecisionTreeClassifier()))
# models.append(('NB', GaussianNB()))
# models.append(('SVM', SVC()))
#
# #evaluate the models
# results = []
# names = []
# #Artificial weights
# pos_p=(train_labels==1)
# neg_p=(train_labels==0)
# ratio = np.sum(neg_p.astype(float))/np.sum(pos_p.astype(float))
# new_trainvec=trainvec
# new_train_labels=train_labels
#
# for j in range(int(ratio-1.0)):
#     new_trainvec=sp.sparse.vstack([new_trainvec,trainvec[pos_p,0::]])
#     new_train_labels=np.concatenate((new_train_labels,train_labels[pos_p]))
#
# for name,model in models:
#     kfold = model_selection.KFold(n_splits=10, random_state=seed)
#     cv_results = model_selection.cross_val_score(model, new_trainvec, new_train_labels, cv=kfold, scoring=scoring)
#     results.append(cv_results)
#     names.append(name)
#     msg = "%s: %f (%f)" %(name, cv_results.mean(), cv_results.std())
#     print(msg)


#The following works with above 60% accuracy
print("Set the training data")
#Artificial weights
pos_p=(train_labels==1)
neg_p=(train_labels==0)
ratio = np.sum(neg_p.astype(float))/np.sum(pos_p.astype(float))
new_trainvec = trainvec
new_train_labels=train_labels
for j in range(int(ratio-1.0)):
    new_trainvec=sp.sparse.vstack([new_trainvec,trainvec[pos_p,0::]])
    new_train_labels=np.concatenate((new_train_labels, train_labels[pos_p]))

print("Fit the classifier")
classifier = SVC(C=0.1,kernel='linear')
classifier.fit(new_trainvec, new_train_labels)

print("Saving the classifier")
#Saving the classifier
file_Name = "classif_all.p"
fileObject = open(file_Name,'wb')
pickle.dump(classifier, fileObject)
fileObject.close()

print ('Validating')

output = classifier.predict(testvec)
clfreport = classification_report(test_labels, output, labels_names=cls_set)
print (clfreport)
print(accuracy_score(test_labels, output)*100)