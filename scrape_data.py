import datetime
import subprocess as sp

import pandas as pd
import praw
import vutils.general as vg

import sqlite3

@vg.timing
def get_posts(subreddit, num_posts: int, output: str):

    post_columns = ['title', 'score', 'upvote_ratio', 'id', 'subreddit', 'url', 'num_comments',
                    'is_reddit_media_domain', 'total_awards_received', 'created']

    posts_df = pd.DataFrame(columns=post_columns)

    for idx, post in enumerate(subreddit.hot(limit=num_posts)):

        posts_df.loc[idx, post_columns] = [post.title, post.score, post.upvote_ratio, post.id,
                                           post.subreddit, post.url, post.num_comments,
                                           post.is_reddit_media_domain, post.total_awards_received,
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

        for idx, comment in enumerate(submission.comments.list()):

            comments_df.loc[idx, comment_columns] = [comment.body, comment.score, post_id,
                                                     comment.total_awards_received, comment.created]

        flist.append(comments_df)

        vg.print_progress(odx, len(id_list), "scraping comments...")

    pd.concat(flist, ignore_index=True).to_csv(output, index=False)


def main():

    client_id = 'gYOYg_jmapcVVZOyEV08pw'
    client_secret = 'VzPlZrSxV6hrzlcrP47nQn-1G9Dliw'
    user_agent = 'generate_data_for_nlp_project'
    username = 'symmetryconserved'
    reddit = praw.Reddit(client_id=client_id, client_secret=client_secret,
                         user_agent=user_agent, username=username)

    # datetime_str = datetime.datetime.now().strftime('%Y_%m_%d-%H_%M_%S')

    # dir_name = f'./data_{datetime_str}'
    dir_name = './data'

    full_cmd = f'mkdir -p {dir_name}'

    sp.run(full_cmd.split())

    subreddit = reddit.subreddit('games')

    # posts_outfile = f'{dir_name}/r_{subreddit.display_name}-posts.csv'
    # comments_outfile = f'{dir_name}/r_{subreddit.display_name}-comments.csv'

    posts_outfile = f'{dir_name}/posts.csv'
    comments_outfile = f'{dir_name}/comments.csv'

    num_posts = 10

    df = get_posts(subreddit=subreddit, num_posts=num_posts, output=posts_outfile)

    get_comments(reddit=reddit, id_list=df.id.values, output=comments_outfile)


if __name__ == "__main__":
    main()
