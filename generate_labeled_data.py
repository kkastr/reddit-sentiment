import re
import enum
import string
import spacy
import numpy as np
import pandas as pd
from tqdm import tqdm
import nltk.sentiment.vader as sev
from spacy.lang.en.stop_words import STOP_WORDS
from vutils.general import print_progress
from params import listOfSubreddits


class LabelMaker(enum.Enum):
    negative = "Negative"
    neutral = "Neutral"
    positive = "Positive"

    @classmethod
    def label(cls, polarity_score):
        if polarity_score <= -0.05:
            return cls.negative.value
        elif polarity_score <= 0.05:
            return cls.neutral.value
        elif polarity_score > 0.05:
            return cls.positive.value


def cleanData(text):
    stopwords = list(STOP_WORDS)
    punctuation = string.punctuation

    doc = nlp(text)
    tokens = []
    clean_tokens = []

    for token in doc:
        if token.lemma_ != "-PRON-":
            tmp = token.lemma_.lower().strip()
        else:
            tmp = token.lower_
        tokens.append(tmp)

    for token in tokens:
        if token not in punctuation and token not in stopwords:
            clean_tokens.append(token)

    return clean_tokens


def createLabeledData(df):

    sia = sev.SentimentIntensityAnalyzer()
    valdict = dict(
        comment_tokens=[],
        title_tokens=[],
        comment_sentiment=[],
        title_sentiment=[],
        comment_labels=[],
        title_labels=[],
    )

    separator = " "
    for idx, row in df.iterrows():

        comment_tokens = cleanData(row.comment_text)
        title_tokens = cleanData(row.post_title)

        comment_sentiment = sia.polarity_scores(separator.join(comment_tokens))["compound"]
        title_sentiment = sia.polarity_scores(separator.join(title_tokens))["compound"]
        comment_label = LabelMaker.label(comment_sentiment)
        title_label = LabelMaker.label(title_sentiment)

        valdict.get("comment_tokens").append(comment_tokens)
        valdict.get("title_tokens").append(title_tokens)
        valdict.get("comment_sentiment").append(comment_sentiment)
        valdict.get("title_sentiment").append(title_sentiment)
        valdict.get("comment_labels").append(comment_label)
        valdict.get("title_labels").append(title_label)

    df["comment_tokens"] = valdict.get("comment_tokens")
    df["title_tokens"] = valdict.get("title_tokens")
    df["comment_sentiment"] = valdict.get("comment_sentiment")
    df["title_sentiment"] = valdict.get("title_sentiment")
    df["comment_labels"] = valdict.get("comment_labels")
    df["title_labels"] = valdict.get("title_labels")

    return df


def main():

    df_list = []

    for subredditName in listOfSubreddits:

        df = pd.read_csv(f"./data/{subredditName}.csv")

        print(f"Processing data from r/{subredditName}...")

        labeled_df = createLabeledData(df)

        print(f"Finished processing r/{subredditName} data.")

        labeled_df.to_csv(f"./data/labeled_{subredditName}.csv", index=False)

        df_list.append(labeled_df)

    cdf = pd.concat(df_list, ignore_index=True)

    cdf.to_csv("./data/combined_labeled_data.csv", index=False)


if __name__ == "__main__":
    nlp = spacy.load(
        "en_core_web_sm",
        enable=["tagger", "attribute_ruler", "lemmatizer"],
        config={"nlp": {"disabled": []}},
    )
    main()
