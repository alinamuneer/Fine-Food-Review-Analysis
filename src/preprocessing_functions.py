"""
Preprocessing functions.
"""
import re
import unicodedata

import spacy
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from contractions import contractions_dict

import pandas as pd
from typing import List, Dict

PATTERNS = [
    re.compile(r"[^\w\s]"),  # punctuation
    re.compile(r"\s{2,}"),  # double spaces
    re.compile(r"&quot;"),  # quote marks
    re.compile(r'@[A-Za-z0-9_]+'),  # mentions
    re.compile(r'https?://[A-Za-z0-9./]+'),  # links
    re.compile(r'#'),  # hashtags
    re.compile(r'\d+'),  # digits
    re.compile(r'((\w)\2{2,})'),  # consecutive repeated characters
    re.compile(r'[^a-zA-z0-9\s]'),  # special characters
    re.compile(r"[´\']s "),
    re.compile(r'&amp')
]


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


def lemmatize(string: str) -> str:
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    doc = nlp(string)
    lemmatized = " ".join([token.lemma_ for token in doc])

    return lemmatized


def remove_stop_words(tokens: List) -> str:
    stop_words = load_stop_words()
    tokens = ' '.join([t for t in tokens if t not in stop_words])
    return tokens


def clean_text(text: str) -> str:
    '''
    Finds patterns in the text and replaces them with sth
    returns a list of words
    '''

    # Remove HTML
    processed_text = remove_html(text)

    # Remove accents
    processed_text = unicodedata.normalize('NFKD', text).encode(
        'ascii', 'ignore').decode('utf-8', 'ignore')

    # Convert to lower case
    processed_text = processed_text.lower()

    # Map contractions to standard forms
    try:
        processed_text = map_contractions(processed_text)
    except:
        pass

    # Clean the text
    processed_text = re.sub(PATTERNS[2], '', processed_text)
    processed_text = re.sub(PATTERNS[3], '', processed_text)
    processed_text = re.sub(PATTERNS[4], '', processed_text)
    processed_text = re.sub(PATTERNS[5], '', processed_text)
    processed_text = re.sub(PATTERNS[6], '', processed_text)
    processed_text = re.sub(PATTERNS[7], r"\2", processed_text)
    processed_text = re.sub(PATTERNS[10], ' ', processed_text)
    processed_text = re.sub(PATTERNS[8], ' ', processed_text)
    processed_text = re.sub(PATTERNS[9], ' ', processed_text)
    processed_text = re.sub(r'_', '', processed_text)

    # Remove punctuation
    processed_text = re.sub(PATTERNS[0], ' ', processed_text)

    # Double spaces converted to one space
    processed_text = re.sub(PATTERNS[1], ' ', processed_text)

    processed_text = lemmatize(processed_text)

    processed_text = word_tokenize(processed_text)

    processed_text = remove_stop_words(processed_text)

    return processed_text