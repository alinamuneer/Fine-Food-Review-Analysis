import pandas as pd
from textblob import TextBlob

def detect_sentiment(text):
    return TextBlob(text).sentiment.polarity

def apply_sentiment_analysis(df : pd.DataFrame) -> pd.DataFrame:
    # Apply the function to each element of the column
    df.summary_cleaned.fillna('', inplace=True)
    df['combined_text_Summary'] = df['text_cleaned'] + ' ' + df['summary_cleaned']
    df['comb_sentiment'] = df.combined_text_Summary.apply(detect_sentiment)
    return df

