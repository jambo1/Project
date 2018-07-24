from textblob import TextBlob
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
import nltk
import ProjectV2.preprocess as prep
import string


wordnet_lemmatizer = WordNetLemmatizer()
tokenizer = RegexpTokenizer(r'\w+')

#Get bigrams from the sentence
def getBigramFeats(features, tokens):
    #Lemmatize the tokens
    lems = [wordnet_lemmatizer.lemmatize(word) for word in tokens]
    bigrams = nltk.bigrams(lems)
    bigrams = [part[0] + ' ' + part[1] for part in bigrams]
    bigramfeat = lems + bigrams

    #Add the bigram features to the features dictionary
    for feat in bigramfeat:
        features['contains(%s)' % feat] = 1.0

#Sentiment features or first and second half of the sentence
def getHalfSentimentFeats(features, tokens):

    if len(tokens) == 1:
        tokens += ['.']
    f_half = tokens[0:int(len(tokens) / 2)]
    s_half = tokens[int(len(tokens) / 2):]

    try:
        blob = TextBlob(
            "".join([" " + i if not i.startswith("'") and i not in string.punctuation else i for i in f_half]).strip())

        features['sentiment fhalf'] = blob.sentiment.polarity
        features['subjective fhalf'] = blob.sentiment.subjectivity

    except:
        features['sentiment fhalf'] = 0.0
        features['subjective fhalf'] = 0.0

    try:
        blob = TextBlob(
            "".join([" " + i if not i.startswith("'") and i not in string.punctuation else i for i in s_half]).strip())

        features['sentiment shalf'] = blob.sentiment.polarity
        features['subjective shalf'] = blob.sentiment.subjectivity

    except:
        features['sentiment shalf'] = 0.0
        features['subjective shalf'] = 0.0

    #Contrast the sentiments
    features['sentiment halfcontrast'] = np.abs(features['sentiment fhalf'] - features['sentiment shalf'])

#Sentiment features from each third of the sentence
def getThirdSentimentFeats(features, tokens):
    # Split sentence into 3rds
    if len(tokens) == 2:
        tokens += ['.']
    f_third = tokens[0:int(len(tokens) / 3)]
    s_third = tokens[int(len(tokens) / 3):2 * int(len(tokens) / 3)]
    t_third = tokens[2 * int(len(tokens) / 3):]

    #Remove punctuation and whitespace and get the sentiment and subjectivity for each third
    try:
        blob = TextBlob(
            "".join([" " + i if not i.startswith("'") and i not in string.punctuation else i for i in f_third]).strip())

        features['sentiment fthird'] = blob.sentiment.polarity
        features['subjective fthird'] = blob.sentiment.subjectivity

    except:
        features['sentiment fthird'] = 0.0
        features['subjective fthird'] = 0.0

    try:
        blob = TextBlob(
            "".join([" " + i if not i.startswith("'") and i not in string.punctuation else i for i in s_third]).strip())

        features['sentiment sthird'] = blob.sentiment.polarity
        features['subjective sthird'] = blob.sentiment.subjectivity

    except:
        features['sentiment sthird'] = 0.0
        features['subjective sthird'] = 0.0

    try:
        blob = TextBlob(
            "".join([" " + i if not i.startswith("'") and i not in string.punctuation else i for i in t_third]).strip())

        features['sentiment tthird'] = blob.sentiment.polarity
        features['subjective tthird'] = blob.sentiment.subjectivity

    except:
        features['sentiment tthird'] = 0.0
        features['subjective tthird'] = 0.0

    #Contrast the sentiments
    features['sentiment 1/2contrast'] = np.abs(features['sentiment fthird'] - features['sentiment sthird'])
    features['sentiment 1/3contrast'] = np.abs(features['sentiment fthird'] - features['sentiment tthird'])
    features['sentiment 2/3contrast'] = np.abs(features['sentiment sthird'] - features['sentiment tthird'])

#POS features of the sentence
def getPOSfeats(features, tokens):
    #Tokenize and lower case the words in sentence
    tokens = [tok.lower() for tok in tokens]

    #POS the tokens and make vectors to count the number of each of the categories below
    pos_vector = nltk.pos_tag(tokens)
    vector = np.zeros(4)
    for i in range(len(pos_vector)):
        pos = pos_vector[i][1]
        # Noun, singular or mass
        if pos[0:2] == 'NN':
            vector[0] += 1
        #Adjective
        elif pos[0:2] == 'JJ':
            vector[1] += 1
        #Verb
        elif pos[0:2] == 'VB':
            vector[2] += 1
        #Adverb
        elif pos[0:2] == 'RB':
            vector[3] += 1

    for j in range(len(vector)):
        features['POS' + str(j + 1)] = vector[j]

#Uses hot ones, i.e. returns 1 if there are more than 4 uppercase letters and 0 if not
def getCapitalFeats(features, sentence):
    count = 0
    threshold = 4
    for j in range(len(sentence)):
        for letter in sentence[j]:
            count += int(letter.isupper())
    features['Capital'] = int(count >= threshold)

#Exclamation mark counts
def getPunctuationCnt(features, sentence):
    count = 0
    punctuation = ['.', '?', '!', '*']
    for i in range(len(sentence)):
        count += int(sentence[i] == '!')
        count += int(sentence[i] == '?')

    features['exclamation'] = count

#Using preprocess emotions the emoticons can be counted
def countEmotion(features, sentence):
    returnsent = sentence
    happy = 0
    sad = 0
    for i in prep.emo_repl:
        if i == 'good':
            happy += returnsent.count(i)
        else:
            sad += returnsent.count(i)

    features['happyemo'] = happy
    features['sademo'] = sad



#Get the topics from the sentence
def getTopicFeats(features, sentence, topic_modeler):
    topics = topic_modeler.transform(sentence)
    for i in range(len(topics)):
        features['Topic :' + str(topics[i][0])] = topics[i][1]

#Number of lol type phrases in the sentence
#Not using at present because such a small portion of the sample have these, is it worthwhile?
def getLOLs(features, tokens):
    laughing = ["lmao", "lmfao", "pmsl", "rofl", "lol"]
    count = 0
    for word in tokens:
        if word in laughing:
            count += 1
    features['lols']=count

#Return all the above
def getallfeatureset(sentence, topic_modeler):
    features = {}
    blob = TextBlob(sentence)
    getCapitalFeats(features, sentence)
    getPunctuationCnt(features, sentence)
    countEmotion(features, sentence)
    getBigramFeats(features, blob.words)
    getHalfSentimentFeats(features, blob.words)
    getThirdSentimentFeats(features, blob.words)
    getPOSfeats(features, blob.words)
    getTopicFeats(features, sentence, topic_modeler)
    #getLOLs(features, blob.words)
    return features

