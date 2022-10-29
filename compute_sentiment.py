import re
import spacy
import string
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report, roc_curve
from sklearn.svm import LinearSVC
from vutils.general import print_progress
from params import listOfSubreddits


def main():

    df = pd.read_csv("./data/combined_labeled_data.csv")

    print(df.groupby("comment_labels").count().comment_text)
    df['comment_labels'] = LabelEncoder().fit_transform(df['comment_labels'])
    x = df["comment_tokens"]
    y = df["comment_labels"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    hyperparams = dict(
        penalty="l1", loss="squared_hinge", dual=False, tol=1e-4, max_iter=1000
    )

    vectorizer = TfidfVectorizer()
    classifier = LinearSVC(**hyperparams)

    steps = [("vectorizer", vectorizer), ("classifier", classifier)]

    pipeln = Pipeline(steps)

    pipeln.fit(x_train, y_train)

    y_pred = pipeln.predict(x_test)

    print(classification_report(y_test, y_pred))
    print("\n\n")
    print(confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    nlp = spacy.load("en_core_web_sm")

    main()
