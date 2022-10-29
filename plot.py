import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import datetime
import os

fontsize = 16

os.system("mkdir -p ./huggingface_plots/")

plt.rc("font", family="serif", size=fontsize)
plt.rc("lines", linewidth=4, aa=True)
plt.rc("figure", figsize=(10, 6))

df = pd.read_csv("huggingface_sentiment_data.csv")

colors = iter(list(mcolors.TABLEAU_COLORS))

hist_kwargs = dict(kde=True, stat="density", alpha=0, edgecolor=None)

df.comment_time = pd.to_datetime(df.comment_time, unit="s")

df.post_time = pd.to_datetime(df.post_time, unit="s")

df["timedelta"] = [item.seconds / 60 / 60 for item in df.comment_time - df.post_time]

df["timeofday"] = [item.hour for item in df.comment_time]

df["slabel"] = [
    "Positive" if v > 0.1 else "Negative" if v < -0.1 else "Neutral" for v in df.comment_sentiment
]

df["hrafter"] = np.round(df.timedelta.values)

subreddit_order = ["games", "news", "politics", "science", "space"]

ax = sns.kdeplot(
    x=df.comment_sentiment, hue=df.subreddit, hue_order=subreddit_order, cut=0, common_norm=False
)
sns.move_legend(ax, "upper left", bbox_to_anchor=(1.04, 1), frameon=False)
ax.set_ylabel("Comment Density")
ax.set_xlabel("Sentiment Value")
plt.tight_layout()
plt.savefig("./huggingface_plots/comment_density_vs_sent_value.png", dpi=600)
plt.show()

ax = sns.kdeplot(
    x=df.timeofday, hue=df.subreddit, hue_order=subreddit_order, cut=0, common_norm=False
)
sns.move_legend(ax, "upper left", bbox_to_anchor=(1.04, 1), frameon=False)
ax.set_ylabel("Comment Density")
ax.set_xlabel("Time of Day (hours)")
plt.tight_layout()
plt.savefig("./huggingface_plots/submission_density_per_time.png", dpi=600)
plt.show()

gb = df.groupby("subreddit").slabel.value_counts(normalize=True).rename("percent").reset_index()
fgrid = sns.catplot(
    x="slabel",
    y="percent",
    hue="subreddit",
    hue_order=subreddit_order,
    order=["Negative", "Neutral", "Positive"],
    kind="bar",
    data=gb,
)
fgrid.axes[0, 0].set_ylabel("Percent of Total")
fgrid.axes[0, 0].set_xlabel("")
plt.savefig("./huggingface_plots/sentiment_percent_per_sub.png", dpi=600)
plt.show()

gb = df.groupby(["timeofday", "subreddit"]).comment_sentiment.mean().rename("avS").reset_index()
ax = sns.lineplot(x="timeofday", y="avS", hue="subreddit", hue_order=subreddit_order, data=gb)
sns.move_legend(ax, "upper left", bbox_to_anchor=(1.04, 1), frameon=False)
ax.set_ylabel("Average Sentiment")
ax.set_xlabel("Time of Day (hours)")
plt.tight_layout()
plt.savefig("./huggingface_plots/average_sentiment_per_time.png", dpi=600)
plt.show()

ax = sns.kdeplot(
    x="timedelta", hue="subreddit", hue_order=subreddit_order, data=df, cut=0, common_norm=False
)
sns.move_legend(ax, "upper left", bbox_to_anchor=(1.04, 1), frameon=False)
ax.set_ylabel("Comment Density")
ax.set_xlabel("Hours After Post Submission")
plt.tight_layout()
plt.savefig("./huggingface_plots/comment_density_v_time_after_submission.png", dpi=600)
plt.show()

gb = df.groupby(["hrafter", "subreddit"]).comment_sentiment.mean().rename("avS").reset_index()
ax = sns.lineplot(x="hrafter", y="avS", hue="subreddit", hue_order=subreddit_order, data=gb)
sns.move_legend(ax, "upper left", bbox_to_anchor=(1.04, 1), frameon=False)
ax.set_ylabel("Average Sentiment")
ax.set_xlabel("Hours After Post Submission")
plt.tight_layout()
plt.savefig("./huggingface_plots/average_sentiment_per_hr_after.png", dpi=600)
plt.show()
