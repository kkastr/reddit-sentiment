import re
import nltk
import nltk.sentiment.vader as sev
import numpy as np
import pandas as pd


# pre-trained sentiment analysis
# nltk.download('vader_lexicon', download_dir='./')

listOfSubreddits = ['games', 'news', 'science', 'space', 'politics', 'gonewild']

sia_cols = ['comment_sentiment', 'title_sentiment', 'comment_score', 'post_score',
            'post_id', 'subreddit']

sia = sev.SentimentIntensityAnalyzer()

df_list = []

for subredditName in listOfSubreddits:

    pf = pd.read_csv(f'./data/{subredditName}_posts.csv')

    df = pd.read_csv(f'./data/{subredditName}_comments.csv')

    gb = df.groupby('post_id')

    for key, grp in gb:
        tdf = pd.DataFrame(columns=sia_cols)

        for item in grp.iterrows():
            post = pf[pf.id == key]

            idx = item[0]
            comment = item[1]
            title_sentiment = sia.polarity_scores(post.title.values[0])['compound']
            sentiment_score = sia.polarity_scores(comment.body)['compound']

            tdf.loc[idx, sia_cols] = [sentiment_score, title_sentiment, comment.score,
                                      post.score.values[0], key, subredditName]
        df_list.append(tdf)

    print(f'Finished processing data from r/{subredditName}.')

cdf = pd.concat(df_list, ignore_index=True)

cdf.to_csv('sentiment_data.csv', index=False)
