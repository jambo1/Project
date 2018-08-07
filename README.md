<h1>Project Purpose</h1>
<p>
  This project is my Masters Development project for the degree of MSc Software Development in the University of Glasgow.
  The purpose of the project is to build a sarcasm analysis algorithm which is then showcased in an educational website
  with the aim of teaching people more about natural language processing. The website includes multiple classifiers to show 
  users how using a different sized sample set to train the algorithm can change the results it produces.
 </p>
<h1>Tools used</h1>
 <p>
    This application has been built with Python, which is the most commonly used machine learning programming language.<br>
    For the front end the web application framework Flask was used along with CSS, HTML and JavaScript with it all
    hosted on PythonAnywhere.<br>
    For the back end TextBlob was utilised to tokenize sentences and to analyse their subjectivity and sentiment.<br>
    NLTK was used to stem and lemmatize sentences, to remove stopwords and to extract bigrams.<br>
    Numpy was used to build and structure the vectors and to store the processed comment files.<br>
    SciPy was used for forming the feature vectors.<br>
    Gensim was used for generating the topics.<br>
    Sklearn was used for saving and loading the topics, the DictVectorizer was used to convert the features to
    a vector dictionary and its SVC classifier was used as the model.<br>
  </p>
