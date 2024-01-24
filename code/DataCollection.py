# Imports
import tweepy
import pandas as pd

# Declare AUTH tokens and keys to access Tweepy
ACCESS_TOKEN = "*****************************"
ACCESS_SECRET = "*****************************"
CONSUMER_KEY = "*****************************"
CONSUMER_SECRET = "*****************************"

# Access Tweepy API
def connect_to_twitter_OAuth():
    auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)

    api = tweepy.API(auth, wait_on_rate_limit=True)
    return api

# Creating an API Object
api = connect_to_twitter_OAuth()

# Declaring create_dataset function
def create_dataset (search_words):
    tweets = tweepy.Cursor(api.search, q=search_words, lang = "en", geo="21.1458°N, 79.0882°E, 1607km", tweet_mode="extended").items(3000)

    df = pd.DataFrame(columns=['text', 'created_at', 'source', 'retweets'])
    for tweet in tweets:
        text = tweet.full_text # utf-8 text of tweet
        created_at =  tweet.created_at # utc time tweet created
        source = tweet.source # utility used to post tweet
        retweets = tweet.retweet_count # number of times this tweet retweeted
        df = df.append({
                'text': tweet['full_text'],
                'created_at': tweet['created_at'],
                'source': tweet['source'],
                'retweets': tweet['retweet_count']
            }, ignore_index=True)
        
        return df

# Creating df_AgriGoI dataset for the agriculture ministry
df_AgriGoI=create_dataset("#FarmBills2020")
agro_feat2=create_dataset("agriculture")
agro_feat3=create_dataset("@nstomar")
agro_feat4=create_dataset("farmers")

df_AgriGoI=df_AgriGoI.append(agro_feat2)
df_AgriGoI=df_AgriGoI.append(agro_feat3)
df_AgriGoI=df_AgriGoI.append(agro_feat4)

# Cleaning Agriculture Dataset
# removing unnecessary columns
df_AgriGoI.drop(['no.','date','src','rt'],axis=1,inplace=True)

# removing retweets
#df_AgriGoI = df_AgriGoI[~df['text'].astype(str).str.startswith('RT')]

#removing unnecessary columns
df_AgriGoI.drop(['no.','date','src','rt'],axis=1,inplace=True)

import re
for i in range(0, len(df_AgriGoI)):
    # Remove all the special characters
    df_AgriGoI.iloc[i,0] = re.sub(r'\W', ' ', df_AgriGoI.iloc[i,0])

    # remove all single characters
    df_AgriGoI.iloc[i,0]= re.sub(r'\s+[a-zA-Z]\s+', ' ', df_AgriGoI.iloc[i,0])

    # Remove single characters from the start
    df_AgriGoI.iloc[i,0] = re.sub(r'\^[a-zA-Z]\s+', ' ', df_AgriGoI.iloc[i,0])

    # Substituting multiple spaces with single space
    df_AgriGoI.iloc[i,0] = re.sub(r'\s+', ' ', df_AgriGoI.iloc[i,0], flags=re.I)

    # Removing prefixed 'b'
    df_AgriGoI.iloc[i,0] = re.sub(r'^b\s+', '', df_AgriGoI.iloc[i,0])

    # Converting to Lowercase
    df_AgriGoI.iloc[i,0] = df_AgriGoI.iloc[i,0].lower()

df_AgriGoI.to_csv("Agriculture_Tweets.csv")