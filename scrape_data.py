import datetime
import re
import subprocess as sp

import pandas as pd
import praw

client_id = 'gYOYg_jmapcVVZOyEV08pw'
client_secret = 'VzPlZrSxV6hrzlcrP47nQn-1G9Dliw'
user_agent = 'generate_data_for_nlp_project'
username = 'symmetryconserved'
reddit = praw.Reddit(client_id = client_id, client_secret = client_secret, user_agent = user_agent, username = username)


post_columns = ['title', 'score', 'upvote_ratio', 'id', 'subreddit', 'url', 'num_comments', 'is_reddit_media_domain', 'total_awards_received' , 'created']

comment_columns = ['body', 'score', 'post_id', 'total_awards_received', 'created']

datetime_str = datetime.datetime.now().strftime(f'%Y_%m_%d-%H_%M_%S')

dir_name = f'./data_{datetime_str}'

full_cmd = f'mkdir -p {dir_name}'

sp.run(full_cmd.split())

subreddit = reddit.subreddit('games')

posts_outfile = f'{dir_name}/r_{subreddit.display_name}-posts.csv'
comments_outfile = f'{dir_name}/r_{subreddit.display_name}-comments.csv'

nposts = 20

posts_df = pd.DataFrame(columns=post_columns)
comments_df = pd.DataFrame(columns=comment_columns)

for idx, post in enumerate(subreddit.hot(limit=nposts)):

    posts_df.loc[idx, post_columns] = [post.title, post.score, post.upvote_ratio, post.id,
                                post.subreddit, post.url, post.num_comments,
                                post.is_reddit_media_domain, post.total_awards_received,
                                post.created]

posts_df.to_csv(posts_outfile, index = False)

for odx, post_id in enumerate(posts_df.id.values):

    submission = reddit.submission(id = post_id)  # submission.comments.replace_more(limit=None)    removes the "more comments" button that hides some of the comments in a thread.

    submission.comments.replace_more(limit = None)

    # submission.comments.list() method does a breadth first traversal of a tree and yields all comments + replies.

    for idx, comment in enumerate(submission.comments.list()):

        comments_df.loc[odx+idx, comment_columns] = [comment.body, comment.score, post_id,
                                            comment.total_awards_received, comment.created]

comments_df.to_csv(comments_outfile, index = False)
