# Our 2020 Election Prediction Results are Here!

We analyzed political trends on Twitter in 7 purported swing states: Arizona, Iowa, Florida, Georgia, Ohio, Texas, and North Carolina. Over the last 10 days, we mined close to 700,000 political tweets -- each geotagged to one of the seven above-mentioned states -- and used machine learning to assign a political label (Democratic/Republican) to each tweet.

We aggregated the tweets from each state to determine our per-state predictions. The predictions for these states are shown in the map below:


Due to the Twitter API rate limitations, we can only make predictions for 7 states; for the remaining 43 states, we use the FiveThirtyEight predictions (https://projects.fivethirtyeight.com/2020-election-forecast/). FiveThirtyEight reports at least 80% confidence in each of the 43 remaining states. Our full election prediction map is shown below:

# Predicted Winner: Joe Biden (420 D / 180 R)

## For more interested readers, please continue reading!

### Data Collection

We collected nearly 700,000 tweets by sampling from the 7 chosen states proportional to each state's population. One consequence of this sampling procedure is that more densely populated states will account for a larger proportion of the 700,000 tweets. However, in general, the densely populated states will have more electoral college votes, so with this sampling procedure the most *important* swing states will have the most *robust* predictions.

To ensure that the collected tweets pertained to the US election, we queried Twitter to match a set of hashtags and keywords related to the US election. The set of keywords/hashtags can be found in supp/hashtags_keywords.txt.

### Machine Learning for Political Label Prediction

**The prediction task is simple: Given a tweet, is the tweet in support of the Democratic party (D) or the Republican party (R)?**

Before we could perform any machine learning on the collected Twitter data, we first needed a way to heuristically annotate a subset of the data. To achieve this, we created two hashtag lists: *hashtag_list_d* and *hashtag_list_r*. Each hashtag list contains a subset of the querying hashtags which clearly indicate support for the Democratic and Republican parties, respectively. We then assign an R label to every tweet which contains a hashtag in *hashtag_list_r*, but not *hashtag_list_d*, and similarly assign a D label to every tweet which contains a hashtag in *hashtag_list_d*, but not *hashtag_list_r*. This gave us an annotated dataset of ~250,000 tweets.
