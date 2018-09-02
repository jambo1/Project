import ProjectV2.get_texts as GetTexts
from ProjectV2.comment import Comment
import ProjectV2.topic_build as topic
import ProjectV2.feature_extract as feature_extract
import os
from sklearn.externals import joblib
import numpy as np

topic_mod = topic.topic(model=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'WebSite/app/SarcFiles/25kFiles/topics.tp'),\
                        dicttp=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'WebSite/app/SarcFiles/25kFiles/topics_dict.tp'))

print(topic_mod.transform("That's life"))

