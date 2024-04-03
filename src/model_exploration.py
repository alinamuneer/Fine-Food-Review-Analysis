from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from src.sentiment_exploration import *


def apply_vectorization(df):
    # Initialize the vectorizer
    vectorizer = CountVectorizer()
    # Fit the vectorizer to the text
    X = vectorizer.fit_transform(df['combined_text_Summary'])
    return X

def apply_naive_bayes(X, df):
    # Initialize the classifier
    nb = MultinomialNB()
    # Fit the classifier
    nb.fit(X, df['Score'])
    return nb

def main():
    # Load your DataFrame here
    df = pd.read_csv("../data/Reviews_processed_1000.csv")
    
    # Apply the parse_and_join function to the DataFrame
    df = apply_parse_and_join(df)

    # Apply the sentiment analysis function to the DataFrame
    df = apply_sentiment_analysis(df)
    
    # Apply the vectorization function to the DataFrame
    X = apply_vectorization(df)
    
    # Apply the naive bayes function to the DataFrame
    nb = apply_naive_bayes(X, df)
    #y_pred_class = nb.predict(X_new)
    #return y_pred_class


if __name__ == "__main__":
    main()