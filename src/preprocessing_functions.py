"""
Preprocessing functions.
"""
import re
import unicodedata

import nltk
import spacy
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from contractions import contractions_dict

import pandas as pd
from typing import List, Dict, Pattern

nltk.download('stopwords')

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

REGEX_PATTERNS: Dict[str, Pattern] = {
    "urls_mentions_hashtags": re.compile(r'https?://[A-Za-z0-9./]+|@[A-Za-z0-9_]+|#'),  # URLs, mentions, and hashtags
    "numeric_values": re.compile(r'\d+'),  # Digits
    "consecutive_char_repetition": re.compile(r'((\w)\2{2,})'),  # Consecutive character repetitions
    "miscellaneous_text_patterns": re.compile(r"&quot;|&amp|[^a-zA-Z0-9\s]|[´\']s "),  # Miscellaneous text patterns
    "punctuation_special_chars": re.compile(r"[^\w\s]|_"),  # Punctuation, special characters, and underscores
    "whitespace_redundancies": re.compile(r"\s{2,}"),  # Redundant whitespaces
}

replacements = {
    "urls_mentions_hashtags": '',  # Removed (substituted with empty string)
    "numeric_values": ' ',  # Replaced with a space
    "consecutive_char_repetition": r"\2",  # Replaced with the second character in the repetition group
    "miscellaneous_text_patterns": ' ',  # Replaced with a space
    "punctuation_special_chars": ' ',  # Replaced with a space
    "whitespace_redundancies": ' ',  # Replaced with a space
}

def add_polarity_label(df: pd.DataFrame) -> pd.DataFrame:
    df['Polarity'] = df.Score.apply(lambda x: 0 if x == 1 or x == 2 else 1)
    return df

def load_stop_words()-> List[str]:
    stop_words: List[str] = stopwords.words('english')
    stop_words.remove('not')
    stop_words.remove('no')

    extra_stop_words: List[str] = ['I']
    stop_words.extend(extra_stop_words)
    return stop_words


def remove_html(text: str) -> str:
    soup = BeautifulSoup(text, 'html.parser')
    cleaned_text = soup.get_text()

    return cleaned_text


def custom_contractions() -> Dict[str, str]:
    contractions_keys = [contr.lower() for contr in contractions_dict.keys()]
    contractions_keys = [re.sub(r"´", r"\'", key) for key in contractions_keys]

    contractions_customed = {contracted: expanded for contracted, expanded in
                             zip(contractions_keys,
                                 contractions_dict.values())}

    return contractions_customed


def map_contractions(text: str) -> str:
    contraction_patterns = [
        re.compile(r"\b\w+[\'|'´]\w+\b", flags=re.IGNORECASE | re.DOTALL),
        re.compile(r'gonna|wanna', flags=re.IGNORECASE | re.DOTALL)]

    matched = re.findall(contraction_patterns[0], text)

    if not matched:
        matched2 = re.findall(contraction_patterns[1], text)
        if not matched2:
            return text
        else:
            expanded_text = re.sub(contraction_patterns[1],
                                   custom_contractions()[matched[0]], text)
            return expanded_text
    else:
        expanded_text = re.sub(contraction_patterns[0],
                               custom_contractions()[matched[0]], text)
        matched2 = re.findall(contraction_patterns[1], expanded_text)

        if not matched2:
            return expanded_text
        else:
            expanded_text2 = re.sub(contraction_patterns[1],
                                    custom_contractions()[matched[0]],
                                    expanded_text)
            return expanded_text2


def lemmatize(string: str, nlp_model) -> str:
    
    doc = nlp_model(string)
    lemmatized = " ".join([token.lemma_ for token in doc])

    return lemmatized

stop_words = set(load_stop_words())

def remove_stop_words(tokens: List, stop_ws) -> str:
    # stop_words = load_stop_words()
    tokens = ' '.join([t for t in tokens if t not in stop_ws])
    return tokens

def apply_regex_patterns(text: str, patterns: Dict[str, Pattern], replacements: Dict[str, str]) -> str:
    """
    Applies specified regex patterns to the text.

    Args:
        text (str): The input text to be processed.
        patterns (Dict[str, Pattern]): A dictionary of compiled regex patterns.
        replacements (Dict[str, str]): A dictionary of replacement strings corresponding to each pattern.

    Returns:
        str: The processed text after applying all regex patterns.
    """
    for pattern_name, pattern in patterns.items():
        replacement = replacements.get(pattern_name, '')  # Get replacement string for the current pattern
        text = pattern.sub(replacement, text)
    return text

def clean_text(text: str) -> str:
    '''
    Finds patterns in the text and replaces them with sth
    returns a list of words
    '''

    # Remove HTML
    processed_text = remove_html(str(text))

    # Remove accents
    processed_text = unicodedata.normalize('NFKD', processed_text).encode(
        'ascii', 'ignore').decode('utf-8', 'ignore')

    # Convert to lower case
    processed_text = processed_text.lower()

    # Map contractions to standard forms
    try:
        processed_text = map_contractions(processed_text)
    except:
        pass

    # Clean the text
    processed_text = apply_regex_patterns(processed_text, REGEX_PATTERNS, replacements)

    processed_text = lemmatize(processed_text, nlp)
    processed_text = word_tokenize(processed_text)
    processed_text = remove_stop_words(processed_text, stop_words)

    return processed_text