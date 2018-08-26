""" This class is a wrapper around the gensim LDA topic modeler. """

from gensim import corpora, models
import nltk
import app.preprocess as preprocess
from nltk.corpus import stopwords

#Class for building the topics
class topic(object):

    #Constructor. Topic number set to 100, no default model
    def __init__(self, nbtopic=100, alpha=1, model=None, dicttp=None):
        self.nbtopic = nbtopic
        self.porter = nltk.PorterStemmer()
        self.alpha = alpha
        self.stop = stopwords.words('english') + ['.', '!', '?', '"', '...', '\\', "''", '[', ']', '~', "'m", "'s", ';',
                                                  "'d",':', '..', '$', "'re", "n't", "'ll"]
        if model != None and dicttp != None:
            self.lda = models.ldamodel.LdaModel.load(model)
            self.dictionary = corpora.Dictionary.load(dicttp)

    #Fitting the comments to the gensim semantics model using Latent Dirichlet Allocation (LDA
    def fit(self, comments):
        tokens = [comment.words for comment in comments]
        #Uniform the words as lower case and stems for the words in each token and remove stopwords
        tokens = [[self.porter.stem(word.lower()) for word in sentence if word.lower() not in self.stop] for sentence in tokens]
        self.dictionary = corpora.Dictionary(tokens)

        #Remove any words appearing less than 10 times
        self.dictionary.filter_extremes(no_below=10)

        #Convert the dictionary to a bag of words
        corpus = [self.dictionary.doc2bow(text) for text in tokens]

        #Feed the data to the LDA model to get the topic data
        self.lda = models.ldamodel.LdaModel(corpus, id2word=self.dictionary, num_topics=self.nbtopic, alpha=self.alpha)

        #Save this for later use in an application setting
        self.lda.save('topics.tp')
        self.dictionary.save('topics_dict.tp')

    #Get a topic by number
    def get_topic(self, topic_number):
        return self.lda.print_topic(topic_number)

    #Transform a sentence to a topic
    def transform(self, sentence):
        #Replace regular troublesome parts of a sentence
        sentence_mod = preprocess.replace_reg(sentence)

        #Tokenize the sentence then stem the words and remove stopwords
        tokens = nltk.word_tokenize(sentence_mod)
        tokens = [self.porter.stem(t.lower()) for t in tokens if t.lower() not in self.stop]

        #Convert the tokens to a bag of words
        corpus_sentence = self.dictionary.doc2bow(tokens)

        return self.lda[corpus_sentence]