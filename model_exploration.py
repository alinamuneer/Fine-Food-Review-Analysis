import pandas as pd
import ast
from textblob import TextBlob, Word


# Define a function to parse string representation of list into list and then join the words
def parse_and_join(s):
    # Parse the string representation of list into a list
    word_list = ast.literal_eval(s)
    # Join the words within the list into a single string
    return ' '.join(word_list)
 
def apply_parse_and_join(df):
    # Apply the function to each element of the column
    df['summary_cleaned'] = df['summary_cleaned'].apply(parse_and_join)
    df['text_cleaned_comp'] = df['text_cleaned'].apply(parse_and_join)
    return df

def detect_sentiment(text):
    return TextBlob(text).sentiment.polarity

def apply_sentiment_analysis(df):
    # Apply the function to each element of the column
    df['combined_text_Summary'] = df['text_cleaned'] + ' ' + df['summary_cleaned']
    df['text_sentiment'] = df.text_cleaned.apply(detect_sentiment)
    df['summary_sentiment'] = df.summary_cleaned.apply(detect_sentiment)
    df['comb_sentiment'] = df.combined_text_Summary.apply(detect_sentiment)
    return df

def main():
    # Load your DataFrame here
    df = pd.read_csv("../data/Reviews_processed_1000.csv")
    
    # Apply the parse_and_join function to the DataFrame
    df = apply_parse_and_join(df)

    # Apply the sentiment analysis function to the DataFrame
    df = apply_sentiment_analysis(df)
    
    # Now df contains the cleaned columns
    print(df.head())


if __name__ == "__main__":
    main()
    