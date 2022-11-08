# Sentiment analysis on reddit comments

Sentiment analysis is the process of programmatically extracting sentiment from text using methods from natural language processing (NLP). The primary method employed in the analysis is the characterization of a piece of text by its polarity, a label that indicates if the text was negative, neutral, or positive (there can be more gradations than these three). Sentiment analysis has been used for several years, with great success, for determining brand sentiment, user response, and more.

In this work, we look at the comments from various subreddits, label with a sentiment value, and then train a classifier for conducting sentiment analysis on the comments.

![dashboard](./plots/sentiment_dashboard.png)

![metrics](./plots/metrics_display.png)

## Usage

The code is run in multiple steps:

First, obtain the relevant api keys from reddit ([instructions](https://docs.aws.amazon.com/solutions/latest/discovering-hot-topics-using-machine-learning/retrieve-and-manage-api-credentials-for-reddit-api-authentication.html)) and place them in a file named `api_keys.py` as such:

```python
client_id = 'your-client-id'
client_secret = 'your-client-key'
user_agent = 'your-app-name'
username = 'your-reddit-username'
```

Next, in `params.py`, specify the number of posts and which subreddits you wish to scrape.

After the setup is complete, successively run the following in your terminal:

```bash
python3 scrape_data.py
```

```bash
python3 generate_labeled_data.py
```

```bash
python3 compute_sentiment.py
```

```bash
python3 make_dashboard.py
```

Once you have succesfully ran all the above scripts, figures will have been generated at `./plots/sentiment_dashboard.png` and `./plots/metrics_display.png` displaying a dashboard of plots made by using the model and metrics that quantify the performance of the model after training, respectively.

## Caveats

Currently, the labeling process is done by using VADER from NTLK, which is a model  trained for classifying sentiment from tweets. Ideally a more accurate labeling process should be devised.
