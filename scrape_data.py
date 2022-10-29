import praw
import pandas as pd
import subprocess as sp
from tqdm import tqdm
from api_keys import client_id, client_secret, user_agent, username
from params import num_posts, listOfSubreddits


def getSubmissionIDs(subreddit, num_posts):
    idlist = []
    for post in subreddit.hot(limit=num_posts):
        if post.stickied:
            continue

        idlist.append(post.id)
    return idlist


def getData(reddit, subreddit):

    cols = [
        "comment_text",
        "comment_score",
        "comment_awards",
        "comment_time",
        "post_title",
        "post_score",
        "post_upvote_ratio",
        "post_id",
        "subreddit",
        "post_url",
        "post_awards",
        "post_time",
    ]

    post_ids = getSubmissionIDs(subreddit=subreddit, num_posts=num_posts)

    rows = []
    for pid in post_ids:

        submission = reddit.submission(id=pid)

        # submission.comments.replace_more(limit=0)
        # removes the "more comments" button that hides some of the comments in a thread.
        # limit=None means remove all instances of "more comments"
        submission.comments.replace_more(limit=10)
        # submission.comments.list()
        # method does a breadth first traversal of a tree and yields all comments + replies.
        for comment in tqdm(submission.comments.list()):

            if comment.stickied:
                continue

            data = [
                comment.body,
                comment.score,
                comment.total_awards_received,
                comment.created,
                submission.title,
                submission.score,
                submission.upvote_ratio,
                submission.id,
                subreddit.display_name,
                submission.url,
                submission.total_awards_received,
                submission.created,
            ]

            rows.append(data)

    df = pd.DataFrame(data=rows, columns=cols)

    return df


def main():

    reddit = praw.Reddit(
        client_id=client_id, client_secret=client_secret, user_agent=user_agent, username=username
    )

    dir_name = "./data"

    full_cmd = f"mkdir -p {dir_name}"

    sp.run(full_cmd.split())

    for subredditName in listOfSubreddits:
        print(f"Scraping r/{subredditName}...")
        subreddit = reddit.subreddit(subredditName)

        outfile = f"{dir_name}/{subredditName}.csv"
        df = getData(reddit=reddit, subreddit=subreddit)

        df.to_csv(outfile, index=False)

        print(f"Finished with r/{subredditName}.")


if __name__ == "__main__":
    main()
