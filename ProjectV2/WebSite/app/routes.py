from app import application
from flask import render_template, flash, redirect
from app.forms import SarcasmForm
import numpy as np
import os
from sklearn.externals import joblib
import app.feature_extract as feature_extract
from app.comment import Comment
import app.topic_build as topic

@application.route('/')
@application.route('/index')
def index():
    return render_template('home.html')

@application.route('/about')
def about():
    return render_template('about.html')

@application.route('/portfolio')
def portfolio():
    return render_template('portfolio.html')

@application.route('/sarcasm', methods=['GET', 'POST'])
def detect_sarcasm():
    form = SarcasmForm()
    if form.validate_on_submit():
        #Save the sample size chosen
        chosenFile = form.sample_size.data

        # Load the relevant DictVectorizer and classifier
        file1 = "app/SarcFiles/"+chosenFile+"/vectordict.p"
        file2 = "app/SarcFiles/"+chosenFile+"/classif_all.p"
        vec = joblib.load(file1)
        classifier = joblib.load(file2)

        #Load the relevant topic modeller
        topic_mod = topic.topic(
            model=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'SarcFiles/'+chosenFile+'/topics.tp'), \
            dicttp=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'SarcFiles/'+chosenFile+'/topics_dict.tp'))

        #sentence = form.user_input.data
        sentence = Comment(form.user_input.data)
        feature = feature_extract.getallfeatureset(sentence, topic_mod)
        feature_vec = vec.transform(feature)
        score = classifier.decision_function(feature_vec)[0]
        result = int(round(2.0 * (1.0 / (1.0 + np.exp(-score)) - 0.5) * 100.0))

        # flash("The percentage of sarcasm for '"+ form.user_input.data+ "' was "+str(result))
        if form.more_details.data:
            return render_template('sarcasm.html', form=form, result=result, feature=feature)
        else:
            return render_template('sarcasm.html', form=form, result=result)
    return render_template('sarcasm.html', form=form)
