
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import pandas as pd
from keras import optimizers
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import csv


def readFile(filename):
    sarc_comments = []
    neg_comments = []

    stops = set(stopwords.words('english'))
    # Open CSV file and read each line
    with open(filename, encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)
        wordnet_lem = WordNetLemmatizer()
        porter = PorterStemmer()
        for row in reader:
            sentence = row['comment']
            if len(sentence.split()) > 3:
                clean = []
                for word in word_tokenize(sentence):
                    if word not in stops:
                        # clean.append(wordnet_lem.lemmatize(word))
                        clean.append(porter.stem(word))
                if (row['label'] == '1'):
                    sarc_comments.append(clean)
                elif (row['label'] == '0'):
                    neg_comments.append(clean)
    # If both arrays have elements save both into np arrays so they can be easier accessed for future uses and return both
    if len(sarc_comments) > 0 and len(neg_comments) > 0:
        # Return both sets of comments
        return sarc_comments, neg_comments

    # If only sarcastic comments then save this and return them
    elif len(sarc_comments) > 0:
        # return sarc_comments
        return sarc_comments

    # If only non sarcastic then save and return them
    elif len(neg_comments) > 0:
        # return neg_comments
        return neg_comments

filename = ("MegaSet.csv")
sarcComments, negComments = readFile(filename)

combined = []
for comment in sarcComments:
    combined.append([1, comment])
for comment in negComments:
    combined.append([0, comment])

dataframe = pd.DataFrame(combined, columns=("label", "comment"))

dataframe = dataframe.sample(frac=1).reset_index(drop=True)

train, test = train_test_split(dataframe, test_size=0.2)

print("Length train ",len(train)," Length test ", len(test))

vocab_size = 15000
batch_size = 100

# define Tokenizer with Vocab Size
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(train['comment'])

x_train = tokenizer.texts_to_matrix(train['comment'], mode='tfidf')
x_test = tokenizer.texts_to_matrix(test['comment'], mode='tfidf')

encoder = LabelBinarizer()
encoder.fit(train['label'])
y_train = encoder.transform(train['label'])
y_test = encoder.transform(test['label'])

model = Sequential()
model.add(Dense(500, input_shape=(vocab_size,)))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(250))
model.add(Activation('tanh'))
model.add(Dropout(0.3))
model.add(Dense(100))
model.add(Activation('tanh'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))
model.summary()

sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
adm = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
model.compile(loss='binary_crossentropy',
              optimizer='Adam',
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=1,
                    verbose=1,
                    validation_split=0.1)

score = model.evaluate(x_test, y_test,
                       batch_size=batch_size, verbose=1)

print('Test accuracy:', score[1])

# text_labels = encoder.classes_
#
# for i in range(10):
#     prediction = model.predict(np.array([x_test[i]]))
#     predicted_label = text_labels[np.argmax(prediction[0])]
#     print(test_files_names.iloc[i])
#     print('Actual label:' + test_tags.iloc[i])
#     print("Predicted label: " + predicted_label)