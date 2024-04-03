import unittest
import pandas as pd

import sys
sys.path.append('..') 

from src.preprocessing_functions import (add_polarity_label,
                                         load_stop_words,
                                         remove_html,
                                         custom_contractions,
                                         map_contractions,
                                         nlp,
                                         lemmatize,
                                         stop_words,
                                         remove_stop_words,
                                         REGEX_PATTERNS,
                                         replacements,
                                         apply_regex_patterns,
                                         clean_text)


class TestFunctions(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({'Text': ['awful', 'quite bad', 'not bad',
                                         'I liked it a lot', 'Best in the world'],
                                'Score': [1, 2, 3, 4, 5]})

    def test_add_polarity_label(self):
        df_with_polarity = add_polarity_label(self.df)
        self.assertTrue('Polarity' in df_with_polarity.columns)
    
    def test_load_stop_words(self):
        stop_words = load_stop_words()
        self.assertIn('this', stop_words)
    
    def test_remove_html(self):
        html_text = "<p>Hello World</p>"
        self.assertEqual(remove_html(html_text), "Hello World")
    
    def test_custom_contractions(self):
        contractions = custom_contractions()
        self.assertEqual(contractions["i'm"], "I am")

    def test_map_contractions(self):
        text = "i'm testing this function."
        expanded_text = map_contractions(text)
        self.assertEqual(expanded_text, "I am testing this function.")
    
    def test_lemmatize(self):
        text = "testing"
        lemmatized_text = lemmatize(text, nlp_model=nlp)
        self.assertEqual(lemmatized_text, "test")
    
    def test_remove_stop_words(self):
        tokens = ['this', 'is', 'a', 'test']
        filtered_tokens = remove_stop_words(tokens, stop_ws=stop_words)
        self.assertNotIn('this', filtered_tokens)

    def test_apply_regex_patterns(self):
        text = "This is a test text with @mention, #hashtag, and https://example.com. It has 1234 numbers."
        expected_output = "This is a test text with hashtag and It has numbers "
        processed_text = apply_regex_patterns(text, REGEX_PATTERNS, replacements)
        self.assertEqual(processed_text, expected_output)
    
    def test_clean_text(self):
        text = "This is a test."
        cleaned_text = clean_text(text)
        self.assertEqual(cleaned_text, 'test')