# Our 2020 Election Prediction Results are Here!

We analyzed political trends on Twitter in 7 purported swing states: Arizona, Iowa, Florida, Georgia, Ohio, Texas, and North Carolina. Over the last 10 days, we mined close to 700,000 political tweets -- each geotagged to one of the seven above-mentioned states -- and used machine learning to assign a political label (Democratic/Republican) to each tweet.

We aggregated the tweets from each state to determine our per-state predictions. The predictions for these states are shown in the map below:


Due to the Twitter API rate limitations, we can only make predictions for 7 states; for the remaining 43 states, we use the FiveThirtyEight predictions (https://projects.fivethirtyeight.com/2020-election-forecast/). FiveThirtyEight reports at least 80% confidence in each of the 43 remaining states. Our full election prediction map is shown below:

# Predicted Winner: Joe Biden (420 D / 180 R)

## For more interested readers, please continue reading!

### Data Collection

We collected nearly 700,000 tweets by sampling from the 7 chosen states proportional to each state's population. One consequence of this sampling procedure is that more densely populated states will take up a larger proportion of the 700,000 tweets. However, in general, the densely populated states have more electoral college votes, so this sampling procedure ensures that the most *important* swing states have the most *robust* predictions.

To ensure that the collected tweets pertained to the US election, we queried Twitter to match a set of hashtags and keywords related to the US election. The set of keywords/hashtags can be found in supp/hashtags_keywords.txt.

### Machine Learning for Political Label Prediction

**The prediction task is simple: Given a tweet, is the tweet in support of the Democratic party (D) or the Republican party (R)?**

Before we could perform any machine learning on the collected Twitter data, we first needed a way to heuristically annotate a subset of the data. To achieve this, we created two hashtag lists: *hashtag_list_d* and *hashtag_list_r*. Each hashtag list contains a subset of the querying hashtags which clearly indicate support for the Democratic and Republican parties, respectively. We then assign an R label to every tweet which contains a hashtag in *hashtag_list_r*, but not *hashtag_list_d*, and similarly assign a D label to every tweet which contains a hashtag in *hashtag_list_d*, but not *hashtag_list_r*. This gave us an annotated dataset of ~250,000 tweets. *hashtag_list_d* and *hashtag_list_r* can be found in supp/hashtags_keywords.txt.

As ~400,000 tweets are still unlabelled, we design a machine learning model called *TweetPredict* to make these predictions. *TweetPredict* takes the text content of a tweet as input, and outputs a probability distribution over the two political parties (D/R). Using our heuristically annotated dataset, *TweetPredict* learns to classify tweets as either (D)emocratic or (R)epublican in a fully-supervised manner.

To ensure that the *TweetPredict* model does not simply learn the mapping from hashtag &rarr; political party, we mask all hashtags contained in either *hashtag_list_d* or *hashtag_list_r* as a preprocessing step.

In addition to the tweet content, we also wanted our model to leverage the structure in the Twitter social network to guide its predictions. We thus collected a list of popular Democratic and Republican Twitter accounts and extracted all the followers of these accounts using the Twitter API. Due to rate limitations, we were limited to extracting 30,000,000 followers over all the accounts. We thus chose the following accounts: 
- Democratic: @JoeBiden (11.6M)
- Republican @Mike_Pence (5.4M), @seanhannity (5.3M), @TuckerCarlson (4.2M), @TeamTrump (2.5M)

We then matched the user accounts of the extracted tweets to the followers of these popular accounts. In total, 80.7% of the extracted tweets were posted by a user who follows at least one of the above popular accounts. The intuition is that a user who follows a Democratic account is more likely to post a tweet that is pro-Democratic (and similarly for the Republican party). We revised our *TweetPredict* model to include these Followers Features (FF); the *TweetPredict* model with FF features is called *TweetPredict + FF*. The *TweetPredict + FF* model is shown in the Figure below:

![image](visual/model_image/model_image.PNG)

69% of our labelled data was labelled D. In general, neural networks bias their predictions in favor of the majority class label. Thus, we decided to reweight the loss function so that the R-labelled tweets are penalized more heavily by the model. The resulting model after reweighting (*TwitterPredict + FF + Reweight*) biases slightly in favor of the Republican party, which is preferred because the US Twitter demographic is supposedly more Democratic than the true US demographic (https://www.pewresearch.org/internet/2019/04/24/sizing-up-twitter-users/).

The table of results for our machine learning models is shown below. We first state a non-machine learning baseline which follows the heuristic to always select the majority class label (D).

![image](visual/model_image/table.PNG)

In the near future, we plan to extend our *TweetPredict* model to a Graph Neural Network framework, so that the Twitter network can be better leveraged for prediction.

### Thanks for reading to the end!
