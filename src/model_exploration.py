from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sentiment_exploration import apply_sentiment_analysis
import pandas as pd
import numpy as np
import sys
import xgboost as xgb

sys.path.append('.')
from settings import REVIEW_FILE_PATH

def apply_vectorization(df: pd.DataFrame) -> CountVectorizer:
    # Initialize the vectorizer
    vect = CountVectorizer(ngram_range = (1,2), max_features=10000, min_df = 2, stop_words = 'english')
    return vect


def xgb_classifier(vect : CountVectorizer, df : pd.DataFrame) -> np.ndarray:
    X_text_features = vect.fit_transform(df.combined_text_Summary)
    df[[-1]] = df[['comb_sentiment']] 
    X_sentiment = df[[-1]]
    X = pd.concat([pd.DataFrame(X_sentiment), pd.DataFrame(X_text_features.toarray())], axis=1)
    y = df['Score']-1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBClassifier(objective='multi:softmax', num_class=5, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(classification_report(y_test, predictions))
    return predictions

def main():
    # Load your DataFrame here
    df = pd.read_csv(REVIEW_FILE_PATH)

    # Apply the sentiment analysis function to the DataFrame
    df = apply_sentiment_analysis(df)
    vect = apply_vectorization(df)
    xgb_classifier(vect,df)




if __name__ == "__main__":
    main()