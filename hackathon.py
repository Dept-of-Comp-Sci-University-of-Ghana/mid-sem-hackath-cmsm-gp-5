import tweepy
from textblob import TextBlob
from wordcloud import WordCloud
import pandas as pd
import numpy as np 
import re
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Reading the kaggle dataset
df = pd.read_csv('read.csv')


#Creating function to Clean Tweets
def cleanTxt(text):
    text = re.sub(r'@[A-Za-z0-9]+', '', text)   # Removes @mention
    text = re.sub(r'#', '',text)    # Removing the hashtags
    text= re.sub(r'RT[\s]:+', '', text)  # Removing RT
    text = re.sub(r'[^a-zA-Z\s]', '', text) #Removing all non english characters
    return text



#Droping all null values 
df = df.dropna()


# Applying function to clean the tweets
df['Tweets'] = df['Tweets'].apply(cleanTxt)


#Creating a function to get subjectivity
def getSubjectivity(text):
    return TextBlob(text).sentiment.subjectivity


#Create a function to get polarity
def getPolarity(text):
    return TextBlob(text).sentiment.polarity


#creating new columns(Applying the subjectivity function and polarity funuction)
df['Subjectivity'] = df['Tweets'].apply(getSubjectivity)
df['Polarity'] = df['Tweets'].apply(getPolarity)


#Creating a function for assigning the positive, negative and neutral values
def getAnalysis(score):
    if (score < 0):
        return 'Negative'
    elif (score == 0):
        return 'Neutral'
    else:
        return 'Positive'


#Applying the analysis function  
df['Analysis'] = df['Polarity'].apply(getAnalysis)

df.to_csv('newread.csv')

df = pd.read_csv('newread.csv')
df = df.dropna()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['Tweets'], df['Analysis'], test_size=0.4, random_state=42)

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Train the sentiment analysis model
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate the model
accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)

# Taking out specific companies
amazon = df[df['Company'] == 'Amazon']
apex = df[df['Company'] == 'ApexLegends']
ascreed = df[df['Company'] == 'AssassinsCreed']

# Count the occurrences of each sentiment in each company
amazon_counts = amazon['Analysis'].value_counts()
apex_counts = apex['Analysis'].value_counts()
ascreed_counts = ascreed['Analysis'].value_counts()

# Set the labels and counts for each sentiment
labels = ['Positive', 'Negative', 'Neutral']
amazon_values = [amazon_counts.get('Positive', 0), amazon_counts.get('Negative', 0), amazon_counts.get('Neutral', 0)]
apex_values = [apex_counts.get('Positive', 0), apex_counts.get('Negative', 0), apex_counts.get('Neutral', 0)]
ascreed_values = [ascreed_counts.get('Positive', 0), ascreed_counts.get('Negative', 0), ascreed_counts.get('Neutral', 0)]

# Set up subplots
fig, axs = plt.subplots(1, 3, figsize=(12, 4))

# Plot the bar plots for each company
axs[0].bar(labels, amazon_values)
axs[0].set_title("Sentiment Analysis for Amazon")

axs[1].bar(labels, apex_values)
axs[1].set_title("Sentiment Analysis for ApexLegends")

axs[2].bar(labels, ascreed_values)
axs[2].set_title("Sentiment Analysis for AssassinsCreed")

# Set common labels for the y-axis and adjust layout
fig.text(0.04, 0.5, 'Count', va='center', rotation='vertical')
plt.tight_layout()

# Display the plot
plt.show()

