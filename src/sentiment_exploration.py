import pandas as pd
from textblob import TextBlob

def detect_sentiment(text):
    return TextBlob(text).sentiment.polarity

def apply_sentiment_analysis(df):
    # Apply the function to each element of the column
    df.summary_cleaned.fillna('', inplace=True)
    df['combined_text_Summary'] = df['text_cleaned'] + ' ' + df['summary_cleaned']
    df['comb_sentiment'] = df.combined_text_Summary.apply(detect_sentiment)
    return df

def main():
    # Load your DataFrame here
    df = pd.read_csv("../data/Reviews_processed_10000.csv")

    # Apply the sentiment analysis function to the DataFrame
    df = apply_sentiment_analysis(df)
    
    # Now df contains the cleaned columns
    #print(df.head())


if __name__ == "__main__":
    main()
    