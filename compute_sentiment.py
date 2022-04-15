import re
from turtle import title
import nltk
import nltk.sentiment.vader as sev
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# from wordcloud import WordCloud
from textblob import TextBlob

# pre-trained sentiment analysis
# nltk.download('vader_lexicon', download_dir='./')

# subredditName = 'news'
listOfSubreddits = ['games', 'news', 'science', 'space', 'politics', 'gonewild']

for subredditName in listOfSubreddits:
    datafile = f'./data/{subredditName}_comments.csv'

    df = pd.read_csv(datafile)

    pf = pd.read_csv(f'./data/{subredditName}_posts.csv')

    # df['body'] = df['body'].str.replace("\W", " ", regex=True)
    # df['body'] = df['body'].str.replace("\s+", " ", regex=True)
    # df['body'] = df['body'].str.replace("\s+[a-zA-Z]\s+", " ", regex=True)
    # df['body'] = df['body'].str.replace("\^[a-zA-Z]\s+", " ", regex=True)
    # df['body'] = df['body'].str.lower()

    # pf['title'] = pf['title'].str.replace("\W", " ", regex=True)
    # pf['title'] = pf['title'].str.replace("\s+", " ", regex=True)
    # pf['title'] = pf['title'].str.replace("\s+[a-zA-Z]\s+", " ", regex=True)
    # pf['title'] = pf['title'].str.replace("\^[a-zA-Z]\s+", " ", regex=True)
    # pf['title'] = pf['title'].str.lower()

    sia = sev.SentimentIntensityAnalyzer()

    sia_cols = ['comment_sentiment', 'title_sentiment', 'comment_score', 'post_score', 'post_id']

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


            tdf.loc[idx, sia_cols] = [sentiment_score, title_sentiment, comment.score,
                                      post.score.values[0], key]
        flist.append(tdf)

    cdf = pd.concat(flist, ignore_index=True)

    grp = cdf.groupby('post_id')

    # print(cdf.post_id.unique())
    # for k, g in grp:
    #     plt.hist(g.comment_sentiment, bins=50)
    #     plt.show()


    # plt.hist(cdf.comment_sentiment)
    # plt.show()

    #

    sns.histplot(cdf.comment_sentiment, kde=True, stat='probability', label=subredditName)
    # sns.kdeplot(cdf.comment_sentiment, label=subredditName)
    # sc = grp.std().comment_sentiment

    # st = grp.std().score

    # plt.scatter(cdf.comment_score, cdf.comment_sentiment)
    # plt.show()
    print(f'finished with r/{subredditName}')

    # for key, grp in gb:
    #     text = grp.body

    #     wc = WordCloud().generate(text)

    #     plt.imshow(wc, interpolation='bilinear')
    #     plt.axis('off')
    #     plt.show()
    #     break


plt.legend()
plt.show()
# cav = cdf.groupby('post_id').median()

# print(cav)
# plt.scatter(cav.sentiment, cav.post_score)

# plt.show()
