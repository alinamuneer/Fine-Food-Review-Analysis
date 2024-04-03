"""
Main script for processing the data.
"""
from time import time

from typing import Callable
from pathlib import WindowsPath
import pandas as pd

import sys
sys.path.append('.')

from settings import REVIEW_FILE_PATH, DATA_DIR
from preprocessing_functions import clean_text, add_polarity_label


def processing_pipeline(data_name: WindowsPath,
                        encoding: str,
                        modify_func: Callable[[pd.DataFrame], pd.DataFrame]) -> pd.DataFrame:

    df = pd.read_csv(data_name,
                     encoding=encoding)
    
    # removing duplicates
    df.drop_duplicates(subset=['Text'], inplace=True)

    # add extra polarity label
    df = modify_func(df)
    
    df['text_cleaned'] = df.Text.apply(lambda x: clean_text(x))
    df['summary_cleaned'] = df.Summary.apply(lambda x: clean_text(x))

    return df


def save_processed_data(df: pd.DataFrame,
                        data_name: str) -> None:
    processed_data_path = f'{DATA_DIR}/processed'
    df.to_csv(f'{processed_data_path}/{data_name}_processed.csv',
              index=False, encoding='utf-8')
    
def main(save: bool = True) -> None:

    data_name = str(REVIEW_FILE_PATH).split('\\')[-1].split('.')[0]
    

    df = processing_pipeline(REVIEW_FILE_PATH,
                                   'utf8',
                                   add_polarity_label)

    if save:
        save_processed_data(df, data_name)

if __name__ == '__main__':
    main()