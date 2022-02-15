import re
import nltk
import nltk.sentiment.vader as sev
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# pre-trained sentiment stuff
# nltk.download('vader_lexicon', download_dir='./')

datafile = f'./data/comments.csv'

df = pd.read_csv(datafile)

pf = pd.read_csv(f'./data/posts.csv')

df['body'] = df['body'].str.replace("\W", " ", regex=True)
df['body'] = df['body'].str.replace("\s+", " ", regex=True)
df['body'] = df['body'].str.replace("\s+[a-zA-Z]\s+", " ", regex=True)
df['body'] = df['body'].str.replace("\^[a-zA-Z]\s+", " ", regex=True)
df['body'] = df['body'].str.lower()

pf['title'] = pf['title'].str.replace("\W", " ", regex=True)
pf['title'] = pf['title'].str.replace("\s+", " ", regex=True)
pf['title'] = pf['title'].str.replace("\s+[a-zA-Z]\s+", " ", regex=True)
pf['title'] = pf['title'].str.replace("\^[a-zA-Z]\s+", " ", regex=True)
pf['title'] = pf['title'].str.lower()


sia = sev.SentimentIntensityAnalyzer()

sia_cols = ['sentiment', 'title_sentiment', 'score', 'post_score', 'post_id']

gb = df.groupby('post_id')

flist = []
for key, grp in gb:
    tdf = pd.DataFrame(columns=sia_cols)

    for item in grp.iterrows():
        post = pf[pf.id == key]

        idx = item[0]
        comment = item[1]
        title_sentiment = sia.polarity_scores(post.title.values[0])['compound']
        sentiment_score = sia.polarity_scores(comment.body)['compound']

        tdf.loc[idx, sia_cols] = [sentiment_score, title_sentiment, comment.score, pf[pf.id==key].score.values[0], key]
    flist.append(tdf)

    # plt.scatter(tdf.score, tdf.sentiment)
    # plt.hist(tdf.sentiment)
    # plt.show()

cdf = pd.concat(flist, ignore_index=True)
# plt.hist(cdf.sentiment)
# plt.show()


# plt.scatter(cdf.sentiment, cdf.post_score)
# plt.show()

# cav = cdf.groupby('post_id').median()

# print(cav)
# plt.scatter(cav.sentiment, cav.post_score)

# plt.show()
