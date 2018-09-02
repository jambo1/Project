import numpy as np
import scipy as sp
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn import model_selection
import pickle
import feature_extract as feature_extract
import get_texts as GetTexts
import topic_build as Topical
import comment as comment
from sklearn.externals import joblib

# #These are the csv files
# sarcComments = GetTexts.read_file("Sarc Set.csv")
# negComments = GetTexts.read_file("Non Sarc Set.csv")

#Uncomment the following when the desired data is already in the system
sarcComments = np.load("WebSite/app/SarcFiles/1kFiles/sarccoms.npy")
negComments = np.load("WebSite/app/SarcFiles/1kFiles/negcoms.npy")

print('Number of  sarcastic coms :', len(sarcComments))
print('Number of  non-sarcastic coms :', len(negComments))

print("Topic time")
topic_mod = Topical.topic(nbtopic=200,alpha='symmetric')
topic_mod.fit(np.concatenate((sarcComments,negComments)))

cls_set = ['Non-Sarcastic', 'Sarcastic']
featuresets = []

for com in sarcComments:
    featuresets.append((feature_extract.getallfeatureset(com, topic_mod), cls_set[1]))

for com in negComments:
    featuresets.append((feature_extract.getallfeatureset(com, topic_mod), cls_set[0]))

featuresets = np.array(featuresets)
targets = (featuresets[0::, 1] == 'Sarcastic').astype(int)

vec = DictVectorizer()
featurevec = vec.fit_transform(featuresets[0::, 0])

# Saving the dictionnary vectorizer
file_Name = "vectordict.p"
#fileObject = open(file_Name, 'wb')
joblib.dump(vec, file_Name)
#fileObject.close()

print('Feature splitting')
# Shuffling
order = shuffle(range(len(featuresets)))
targets = targets[order]
featurevec = featurevec[order, 0::]

# Spliting
size = int(len(featuresets) * .3)  # 30% is used for the test set

trainvec = featurevec[size:, 0::]
train_targets = targets[size:]
testvec = featurevec[:size, 0::]
test_targets = targets[:size]

# Artificial weights
pos_p = (train_targets == 1)
neg_p = (train_targets == 0)
ratio = np.sum(neg_p.astype(float)) / np.sum(pos_p.astype(float))
new_trainvec = trainvec
new_train_targets = train_targets
for j in range(int(ratio - 1.0)):
    new_trainvec = sp.sparse.vstack([new_trainvec, trainvec[pos_p, 0::]])
    new_train_targets = np.concatenate((new_train_targets, train_targets[pos_p]))

labels = np.unique(new_train_targets)
print(labels)

# #To update cannot use for SVM
# file1 = open('classif_all.p','rb')
# classifier= pickle.load(file1)
# file1.close()
# classifier.update(new_trainvec, new_train_targets)

#To implement classifier first time
classifier = SVC(C=0.1,kernel='linear')
classifier.fit(new_trainvec, new_train_targets)

# results =[]
# names =[]
# model = GaussianNB()
# kfold = model_selection.KFold(n_splits=10, random_state=7)
# cv_results = model_selection.cross_val_score(model, trainvec.todense(), new_train_targets, cv=kfold, scoring='accuracy')
# results.append(cv_results)
# print(cv_results)
# msg = "%s: %f (%f)" %("NB", cv_results.mean(), cv_results.std())
# print(msg)



# #Saving the classifier
file_Name = "classif_all.p"
# fileObject = open(file_Name, 'wb', )
# pickle.dump(classifier, fileObject)
# fileObject.close()
joblib.dump(classifier, file_Name)


print('Validating SVM')

output = classifier.predict(testvec)
clfreport = classification_report(test_targets, output, target_names=cls_set)
print (clfreport)
print(accuracy_score(test_targets, output) * 100)

#
# models = []
# models.append(('LR', LogisticRegression()))
# models.append(('LDA', LinearDiscriminantAnalysis()))
# models.append(('KNN', KNeighborsClassifier()))
# models.append(('CART', DecisionTreeClassifier()))
# models.append(('NB', GaussianNB()))
# models.append(('SVM', SVC()))
#
# results = []
# names = []
# seed=7
# scoring='accuracy'
# for name,model in models:
#     kfold = model_selection.KFold(n_splits=10, random_state=seed)
#     cv_results = model_selection.cross_val_score(model, trainvec.todense(), new_train_targets, cv=kfold, scoring=scoring)
#     results.append(cv_results)
#     names.append(name)
#     msg = "%s: %f (%f)" %(name, cv_results.mean(), cv_results.std())
#     print(msg)
