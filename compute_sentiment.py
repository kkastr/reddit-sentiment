import os
import spacy
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report, roc_curve
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.svm import LinearSVC
from params import listOfSubreddits


def main():

    fontsize = 16
    os.system("mkdir -p ./plots/")
    plt.rc("font", family="serif", size=fontsize)
    plt.rc("lines", linewidth=4, aa=True)

    fig, axes = plt.subplots(1, 2, figsize=(18, 9))

    df = pd.read_csv("./data/combined_labeled_data.csv")
    df = df.sample(frac=1, random_state=1).reset_index()
    df["comment_labels"] = LabelEncoder().fit_transform(df["comment_labels"])
    x = df["comment_tokens"]
    y = df["comment_labels"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    hyperparams = dict(penalty="l1", loss="squared_hinge", dual=False, tol=1e-4, max_iter=1000)

    vectorizer = TfidfVectorizer()
    classifier = LinearSVC(**hyperparams)

    steps = [("vectorizer", vectorizer), ("classifier", classifier)]

    pipeln = Pipeline(steps)

    pipeln.fit(x_train, y_train)

    y_pred = pipeln.predict(x_test)

    labels_strs = ["Negative", "Neutral", "Positive"]
    clf_report = classification_report(y_test, y_pred, target_names=labels_strs, output_dict=True)

    # classification_report(y_test, y_pred, target_names=label_strs)
    # print("\n\n")
    # print(confusion_matrix(y_test, y_pred))
    start_index = len(df) - len(y_test)
    nf = df.loc[start_index:, df.columns]

    nf.loc[:, "comment_label_prediction"] = y_pred
    nf.loc[:, "comment_label_prediction"] = nf["comment_label_prediction"].map(
        {0: "Negative", 1: "Neutral", 2: "Positive"}
    )

    nf.to_csv("./data/sentiment_classifier_predictions.csv", index=False)

    sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, ax=axes[0], annot=True, cmap="viridis")

    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=labels_strs, normalize="true", ax=axes[1], cmap="Blues"
    )
    plt.tight_layout()

    plt.savefig("./plots/metrics_display.png", dpi=600)


if __name__ == "__main__":
    nlp = spacy.load("en_core_web_sm")

    main()
