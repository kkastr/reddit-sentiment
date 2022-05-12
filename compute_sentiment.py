import re
import nltk
import numpy as np
import pandas as pd
import nltk.sentiment.vader as sev


from params import listOfSubreddits

# pre-trained sentiment analysis
# nltk.download('vader_lexicon', download_dir='./')


def main():
    sia_cols = ['comment_sentiment', 'title_sentiment', 'comment_score', 'post_score',
                'post_upvote_ratio', 'post_id', 'subreddit', 'post_time', 'comment_time']

    sia = sev.SentimentIntensityAnalyzer()

    df_list = []

    for subredditName in listOfSubreddits:

        pf = pd.read_csv(f'./data/{subredditName}_posts.csv')

        cf = pd.read_csv(f'./data/{subredditName}_comments.csv')

        gb = cf.groupby('post_id')

        for key, grp in gb:
            tdf = pd.DataFrame(columns=sia_cols)

            for item in grp.iterrows():
                post = pf[pf.id == key]

                idx = item[0]
                comment = item[1]
                title_sentiment = sia.polarity_scores(post.title.values[0])['compound']
                sentiment_score = sia.polarity_scores(comment.body)['compound']

                output_data = [sentiment_score, title_sentiment, comment.score,
                               post.score.values[0], post.upvote_ratio.values[0],
                               key, subredditName,
                               post.created.values[0], comment.created]

                tdf.loc[idx, sia_cols] = output_data
            df_list.append(tdf)

        print(f'Finished processing data from r/{subredditName}.')

    cdf = pd.concat(df_list, ignore_index=True)

    cdf.to_csv('sentiment_data.csv', index=False)


if __name__ == "__main__":
    main()
