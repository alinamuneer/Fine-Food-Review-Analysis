"""
Main script for processing the data.
"""
from time import time
import re
from pathlib import Path
import unicodedata
import pandas as pd
import os

import spacy
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from contractions import contractions_dict

from settings import *
# DIR_PATH = Path(__file__).parent.parent
N = 1000

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


def load_data(data_name: str, encoding: str) -> pd.DataFrame:
    data_path = os.path.join(data_name)
    df = pd.read_csv(data_path,
                     encoding=encoding)
    return df

def drop_columns(df: pd.DataFrame, columns: list):
    for column in columns:
        df.drop(column, axis=1, inplace=True)
    return df

def add_polarity_label(df: pd.DataFrame) -> pd.DataFrame:
    df['Polarity'] = df.Score.apply(lambda x: 0 if x == 1 or x == 2 else 1)
    return df

def load_stop_words():
    stop_words = stopwords.words('english')
    stop_words.remove('not')
    stop_words.remove('no')

    extra_stop_words = ['I']
    stop_words.extend(extra_stop_words)
    return stop_words


def remove_html(text):
    soup = BeautifulSoup(text, 'html.parser')
    cleaned_text = soup.get_text()

    return cleaned_text


def custom_contractions():
    contractions_keys = [contr.lower() for contr in contractions_dict.keys()]
    contractions_keys = [re.sub(r"´", r"\'", key) for key in contractions_keys]

    contractions_customed = {contracted: expanded for contracted, expanded in
                             zip(contractions_keys,
                                 contractions_dict.values())}

    return contractions_customed


def map_contractions(text):
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


def lemmatize(string):
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    doc = nlp(string)
    lemmatized = " ".join([token.lemma_ for token in doc])

    return lemmatized


def remove_stop_words(tokens):
    stop_words = load_stop_words()
    tokens = [t for t in tokens if t not in stop_words]
    return tokens


def clean_text(text: str):
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

    # Lemmatize
    processed_text = lemmatize(processed_text)

    # Tokenize
    processed_text = word_tokenize(processed_text)

    # Remove stopwords
    processed_text = remove_stop_words(processed_text)

    return processed_text


def processing_pipeline(data_name,
                        encoding,
                        modify_func) -> pd.DataFrame:

    df = load_data(data_name, encoding=encoding)

    df = modify_func(df)
    df = df.sample(n=N, random_state=42)
    
    df['text_cleaned'] = df.Text.apply(lambda x: clean_text(x))
    df['summary_cleaned'] = df.Summary.apply(lambda x: clean_text(x))

    return df


def save_processed_data(df: pd.DataFrame,
                        data_name: str) -> None:
    processed_data_path = 'data/processed'
    df.to_csv(f'{processed_data_path}/{data_name}_processed_{N}.csv',
              index=False, encoding='utf-8')
    
def main(save: bool = True) -> None:

    t = time()
    print('Processing...')
    data_name = str(REVIEW_FILE_PATH).split('\\')[-1].split('.')[0]
    


    # processing dataset
    df = processing_pipeline(REVIEW_FILE_PATH,
                                   'utf8',
                                   add_polarity_label)
    
    elapsed_time = round((time() - t) / 60, 2)
    print(f'Time to clean up everything: {elapsed_time} min.')

    # save
    if save:
        save_processed_data(df, data_name)

if __name__ == '__main__':
    main()