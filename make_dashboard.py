import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import datetime
import os
from params import listOfSubreddits

fontsize = 16

os.system("mkdir -p ./plots/")

plt.rc("font", family="serif", size=fontsize)
plt.rc("lines", linewidth=4, aa=True)

fig, axes = plt.subplots(2, 3, figsize=(18, 9))


df = pd.read_csv("./data/sentiment_classifier_predictions.csv")

colors = iter(list(mcolors.TABLEAU_COLORS))

hist_kwargs = dict(kde=True, stat="density", alpha=0, edgecolor=None)

df.comment_time = pd.to_datetime(df.comment_time, unit="s")

df.post_time = pd.to_datetime(df.post_time, unit="s")

df["timedelta"] = [item.seconds / 60 / 60 for item in df.comment_time - df.post_time]

df["timeofday"] = [item.hour for item in df.comment_time]

df["hrafter"] = np.round(df.timedelta.values)

sentiment_order = ["Negative", "Neutral", "Positive"]
sentiment_colors = ["cadetblue", "silver", "lightcoral"]

hours = np.array([0, 6, 12, 18, 23])
hrlabels = np.array([str(i) for i in hours])
xmin = hours.min() - 1
xmax = hours.max() + 1

barplot_kwargs = dict(
    kind="bar", stacked=True, width=1, edgecolor="k", color=sentiment_colors, rot=0, legend=False
)


top_left, top_mid, top_right = axes[1, 0], axes[1, 1], axes[1, 2]
bot_left, bot_mid, bot_right = axes[0, 0], axes[0, 1], axes[0, 2]


sns.kdeplot(
    ax=top_left,
    x="timedelta",
    hue="subreddit",
    hue_order=listOfSubreddits,
    data=df,
    cut=0,
    common_norm=False,
    legend=False,
)
top_left.set_ylabel("Comment Density")
top_left.set_xlabel("Hours After Post Submission")

top_left.set_xticks(hours)
top_left.set_xticklabels(hrlabels)
top_left.set_xlim([xmin, xmax])

gb = (
    df.groupby("subreddit")
    .comment_label_prediction.value_counts(normalize=True)
    .rename("percent")
    .reset_index()
)

sns.barplot(
    ax=top_mid,
    x="comment_label_prediction",
    y="percent",
    hue="subreddit",
    hue_order=listOfSubreddits,
    order=["Negative", "Neutral", "Positive"],
    edgecolor="k",
    data=gb,
)
top_mid.set_ylabel("Fraction")
top_mid.set_xlabel("")
top_mid.legend([], [], frameon=False)

sns.kdeplot(
    ax=top_right,
    x=df.timeofday,
    hue=df.subreddit,
    hue_order=listOfSubreddits,
    cut=0,
    common_norm=False,
    legend=False,
)
top_right.set_ylabel("Comment Density")
top_right.set_xlabel("Time of Day (hours)")

top_right.set_xticks(hours)
top_right.set_xticklabels(hrlabels)
top_right.set_xlim([xmin, xmax])


gb = df.groupby("comment_label_prediction").comment_label_prediction.count()

donut_hole = plt.Circle((0, 0), 0.7, color="white")
bot_mid.pie(
    gb.values,
    labels=gb.index.values,
    colors=sentiment_colors,
    startangle=30,
    autopct="%1.1f%%",
    pctdistance=0.85,
)

bot_mid.add_artist(donut_hole)

gb = (
    df.groupby(["hrafter", "comment_label_prediction"])
    .comment_label_prediction.count()
    .rename("avS")
    .reset_index()
    .pivot(index="hrafter", columns="comment_label_prediction")
)

rows_to_drop = gb[gb.sum(axis=1) < 100]["avS"].index.values
nf = gb["avS"].loc[:, sentiment_order].div(gb.sum(axis=1), axis="index")

nf.plot(ax=bot_left, xlabel="Hours After Post Submission", **barplot_kwargs)

bot_left.set_xticks(hours)
bot_left.set_xticklabels(hrlabels)
bot_left.set_xlim([xmin, xmax])

gb = (
    df.groupby(["timeofday", "comment_label_prediction"])
    .comment_label_prediction.count()
    .rename("avS")
    .reset_index()
    .pivot(index="timeofday", columns="comment_label_prediction")
)

rows_to_drop = gb[gb.sum(axis=1) < 100]["avS"].index.values
nf = gb["avS"].loc[:, sentiment_order].div(gb.sum(axis=1), axis="index")

nf.plot(ax=bot_right, xlabel="Time of Day (hours)", **barplot_kwargs)
bot_right.set_xticks(hours)
bot_right.set_xticklabels(hrlabels)
bot_right.set_xlim([xmin, xmax])


handles, labels = top_mid.get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    fontsize=12,
    loc="lower center",
    bbox_to_anchor=(0.525, 0),
    ncol=len(listOfSubreddits),
    frameon=False,
    handletextpad=0.5,
    columnspacing=0.5,
)
fig.tight_layout()

plt.savefig("./plots/sentiment_dashboard.png", dpi=600)

plt.show()
