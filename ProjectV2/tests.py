import unittest
import ProjectV2.get_texts as get_texts
import ProjectV2.run_sarcasm as run_sarcasm

class InputValidationTests(unittest.TestCase):

    def test_get_texts_requires_3_alphanum_word_input(self):
        sarc_comments, neg_comments = get_texts.read_file("tests.csv")
        self.assertEqual(len(sarc_comments), 2, "Sarcastic comments array length")
        self.assertEqual(len(neg_comments), 2, "Non-sarcastic comments array length")

    def test_one_word_run_sarcasm_input_isnt_positive(self):
        score = run_sarcasm.getSarcasmScore("a")
        self.assertLess(score, 1, "Doesn't define one word as sarcastic")

