import datetime
import subprocess as sp

import pandas as pd
import praw
import vutils.general as vg

import sqlite3

from api_keys import client_id, client_secret, user_agent, username


@vg.timing
def get_posts(subreddit, num_posts: int, output: str):

    post_columns = ['title', 'score', 'upvote_ratio', 'id', 'subreddit', 'url', 'num_comments',
                    'is_reddit_media_domain', 'total_awards_received', 'created']

    posts_df = pd.DataFrame(columns=post_columns)

    for post in subreddit.hot(limit=num_posts):

        if post.stickied:
            continue

        posts_df.loc[len(posts_df), post_columns] = [post.title, post.score, post.upvote_ratio,
                                                     post.id, post.subreddit, post.url,
                                                     post.num_comments,
                                                     post.is_reddit_media_domain,
                                                     post.total_awards_received,
                                                     post.created]

    posts_df.to_csv(output, index=False)

    return posts_df


@vg.timing
def get_comments(reddit, id_list, output):
    comment_columns = ['body', 'score', 'post_id', 'total_awards_received', 'created']

    flist = []

    for odx, post_id in enumerate(id_list):

        submission = reddit.submission(id=post_id)
        # submission.comments.replace_more(limit=None)
        # removes the "more comments" button that hides some of the comments in a thread.

        submission.comments.replace_more(limit=None)

        # submission.comments.list()
        # method does a breadth first traversal of a tree and yields all comments + replies.

        comments_df = pd.DataFrame(columns=comment_columns)

        for comment in submission.comments.list():

            if comment.stickied:
                continue

            comments_df.loc[len(comments_df), comment_columns] = [comment.body, comment.score,
                                                                  post_id,
                                                                  comment.total_awards_received,
                                                                  comment.created]

        flist.append(comments_df)

        vg.print_progress(odx, len(id_list), "scraping comments...")

    pd.concat(flist, ignore_index=True).to_csv(output, index=False)


def main():

    reddit = praw.Reddit(client_id=client_id, client_secret=client_secret,
                         user_agent=user_agent, username=username)

    # datetime_str = datetime.datetime.now().strftime('%Y_%m_%d-%H_%M_%S')

    # dir_name = f'./data_{datetime_str}'
    dir_name = './data'

    full_cmd = f'mkdir -p {dir_name}'

    sp.run(full_cmd.split())

    listOfSubreddits = ['games', 'news', 'science', 'space', 'politics', 'gonewild']

    num_posts = 10

    for subredditName in listOfSubreddits:

        subreddit = reddit.subreddit(subredditName)

        posts_outfile = f'{dir_name}/{subredditName}_posts.csv'
        comments_outfile = f'{dir_name}/{subredditName}_comments.csv'

        df = get_posts(subreddit=subreddit, num_posts=num_posts, output=posts_outfile)

        get_comments(reddit=reddit, id_list=df.id.values, output=comments_outfile)


if __name__ == "__main__":
    main()
