import unittest
import ProjectV2.get_texts as get_texts
import ProjectV2.run_sarcasm as run_sarcasm
import ProjectV2.topic_build as topic
import ProjectV2.preprocess as prep
from ProjectV2.comment import Comment
import ProjectV2.feature_extract as feature_extract
import os

#Tests to ensure that the rules for training comments are met
class InputValidationTests(unittest.TestCase):
    def test_get_texts_requires_3_alphanum_word_input(self):
        sarc_comments, neg_comments = get_texts.read_file("tests.csv")
        self.assertEqual( 2,len(sarc_comments), "Sarcastic comments array length")
        self.assertEqual( 2,len(neg_comments), "Non-sarcastic comments array length")

#Tests that an input with no real features doesnt give a sarcastic score
class SarcasmScoreTest(unittest.TestCase):
    def test_one_word_run_sarcasm_input_isnt_positive(self):
        score = run_sarcasm.getSarcasmScore("a")
        self.assertGreater( 1,score, "Doesn't define one word input as sarcastic")

    def test_multiple_as_for_sarcasm_not_positive(self):
        score = run_sarcasm.getSarcasmScore("a a a a a a a a a a a a a a")
        self.assertGreater(1, score, "Doesn't define long input as sarcastic")

#Test the topic modeller returns a topic, which is correct, and that the score is also correct
class TopicModellerTest(unittest.TestCase):
    def test_topic_modeller(self):
        topic_mod = topic.topic(
            model=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'WebSite/app/SarcFiles/25kFiles/topics.tp'), \
            dicttp=os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                'WebSite/app/SarcFiles/25kFiles/topics_dict.tp'))

        result = topic_mod.transform("That's life")
        result=result[0]
        self.assertEqual(result[0], 177, "Topic is incorrect")
        self.assertAlmostEqual( 0.5025,result[1], places=4, msg="Topic score is incorrect")

#Test preprocess replaces the correct things with the correct replacement
class PreprocessTest(unittest.TestCase):
    def test_regular_replace1(self):
        sentence1="r u hahaha don't I'll"
        self.assertEqual("are you ha do not I will", prep.replace_reg(sentence1), "Regular replace sentence1 wrong")

    def test_regular_replace2(self):
        sentence2 = "hasn't can't you're would've :)"
        self.assertEqual( "has not can not you are would have smile",prep.replace_reg(sentence2), "Regular replace sentence2 wrong")

    def test_happy_emo_replace(self):
        sentence=":) :') :p"
        self.assertEqual("good good good", prep.replace_emo(sentence), "Happy emoticon replacement is wrong")

    def test_sad_emo_replace(self):
        sentence=":( :s :/"
        self.assertEqual("bad bad bad", prep.replace_emo(sentence), "Sad emoticon replacement wrong")

#Test that the feature extract extracts the correct features from sentences
class FeatureValidationTests(unittest.TestCase):
    def setUp(self):
        topic_mod = topic.topic(
            model=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'WebSite/app/SarcFiles/25kFiles/topics.tp'), \
            dicttp=os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                'WebSite/app/SarcFiles/25kFiles/topics_dict.tp'))
        com1 = Comment("AAAA !?!? lol :) :(")
        self.features1 = feature_extract.getallfeatureset(com1, topic_mod)
        com2= Comment("I hate this horrible place and this plate")
        self.features2=feature_extract.getallfeatureset(com2, topic_mod)
        com3 = Comment("plate red quickly ran")
        self.features3 = feature_extract.getallfeatureset(com3, topic_mod)

    def test_cap_features(self):
        self.assertEqual(1, self.features1['Capital'], "Capitals wrong")

    def test_exclamation_features(self):
        self.assertEqual(4, self.features1['exclamation'], "Punctuation count wrong")

    def test_lol_features(self):
        self.assertEqual(1, self.features1['lols'], "LOLs wrong")

    def test_half_sentiment(self):
        self.assertGreater(0, self.features2['sentiment fhalf'], "first half sentiment wrong")
        self.assertAlmostEqual(0, self.features2['sentiment shalf'], places=3, msg="second half sentiment wrong")

    def test_third_sentiment(self):
        self.assertGreater(0, self.features2['sentiment fthird'], "First third sentiment wrong")
        self.assertGreater(0, self.features2['sentiment sthird'], "Second third sentiment wrong")
        self.assertAlmostEqual(0, self.features2['sentiment tthird'], places=3, msg="Third third sentiment wrong")

    def test_POS(self):
        self.assertEqual(1, self.features3['POS1'], "nouns wrong")
        self.assertEqual(1, self.features3['POS2'], "adjectives wrong")
        self.assertEqual(1, self.features3['POS3'], "verbs wrong")
        self.assertEqual(1, self.features3['POS4'], "adverbs wrong")

    def test_emoticons(self):
        self.assertEqual(1, self.features1['happyemo'], "Happy emoticons wrong")
        self.assertEqual(1, self.features1['sademo'], "Sad emoticons wrong")
