import unittest
import pandas as pd

import sys
sys.path.append('..') 

from src.preprocessing_functions import (add_polarity_label,
                                         load_stop_words,
                                         remove_html,
                                         custom_contractions,
                                         map_contractions,
                                         lemmatize,
                                         remove_stop_words,
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
        lemmatized_text = lemmatize(text)
        self.assertEqual(lemmatized_text, "test")
    
    def test_remove_stop_words(self):
        tokens = ['this', 'is', 'a', 'test']
        filtered_tokens = remove_stop_words(tokens)
        self.assertNotIn('this', filtered_tokens)
    
    def test_clean_text(self):
        text = "This is a test."
        cleaned_text = clean_text(text)
        self.assertEqual(cleaned_text, 'test')

if __name__ == '__main__':
    unittest.main()