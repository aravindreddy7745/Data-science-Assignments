import warnings
warnings.filterwarnings("ignore")

# Download necessary NLTK resources
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')
nltk.download('averaged_perceptron_tagger')
nltk.download('movie_reviews')
nltk.download('punkt')
nltk.download('conll2000')
nltk.download('brown')

# Load the CSV file
import pandas as pd
df = pd.read_csv('Elon_musk.csv', encoding='latin1')

# Basic PreProcessing
df['Tweets'] = df['Text'].str.replace(r'@\w+', '')
tweets = [Text.strip() for Text in df.Tweets]

# Combining all the tweets into a single text
combined_text = ''.join(tweets)

#===================================================================================
# Clean Text
import re
import string
def clean_text(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

cleaned_text = clean_text(combined_text)

#===================================================================================
# Remove emojis
def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

cleaned_text = remove_emoji(cleaned_text)

#===================================================================================
# Preprocess text using NLTK
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

from nltk.tokenize import word_tokenize
def preprocess_text(text):
    text = text.lower()
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    text = ' '.join(tokens)
    return text

df['cleaned_text'] = df['Tweets'].apply(preprocess_text)

#==================================================================================
# Sentiment Analysis
from textblob import TextBlob
def get_textblob_sentiment(cleaned_text):
    analysis = TextBlob(cleaned_text)
    if analysis.sentiment.polarity > 0:
        return 'positive'
    elif analysis.sentiment.polarity < 0:
        return 'negative'
    else:
        return 'neutral'

from nltk.sentiment.vader import SentimentIntensityAnalyzer
def get_vader_sentiment(cleaned_text):
    sid = SentimentIntensityAnalyzer()
    compound_score = sid.polarity_scores(cleaned_text)['compound']
    if compound_score >= 0.05:
        return 'positive'
    elif compound_score <= -0.05:
        return 'negative'
    else:
        return 'neutral'

df['textblob_sentiment'] = df['cleaned_text'].apply(get_textblob_sentiment)
df['vader_sentiment'] = df['cleaned_text'].apply(get_vader_sentiment)

#===================================================================================
# Check for missing values
missing_values = df.isnull().sum()
missing_values_column = df['Text'].isnull().sum()

# Print missing values
print("Missing Values in the Entire Dataset:")
print(missing_values)

print("\nMissing Values in a Specific Column:")
print(missing_values_column)

# Explore data distribution of sentiment labels
sentiment_distribution = df['vader_sentiment'].value_counts()

# Plotting the distribution
import matplotlib.pyplot as plt
sentiment_distribution.plot(kind='bar', color=['green', 'red', 'blue'])
plt.title('Distribution of Sentiment Labels')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()

#===================================================================================
# Split the data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(df['cleaned_text'], df['vader_sentiment'], test_size=0.2, random_state=42)

#===================================================================================
# Text Vectorization
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

#==================================================================================
# Train a Sentiment Analysis Model
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train_tfidf, Y_train)
predictions = model.predict(X_test_tfidf)

#=================================================================================
# Evaluate accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(Y_test, predictions)
print(f"Accuracy: {accuracy}")                      # Accuracy: 0.755

#=================================================================================
# Generate WordCloud
from wordcloud import WordCloud
all_text = ' '.join(df['cleaned_text'])
wordcloud = WordCloud(width=800, height=400, random_state=42, max_font_size=100, background_color='white').generate(all_text)

# Plot WordCloud image
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

#==================================================================================
# Sentiment Analysis for New Text
new_text = "I really enjoyed the movie. It was fantastic!"
new_text_processed = preprocess_text(new_text)
new_text_vectorized = vectorizer.transform([new_text_processed])
prediction = model.predict(new_text_vectorized)[0]
print(f"Predicted Sentiment: {prediction}")

# Distribution of Text Lengths:
df['text_length'] = df['cleaned_text'].apply(lambda x: len(x.split()))
plt.figure(figsize=(10, 6))
plt.hist(df['text_length'], bins=30, color='skyblue', edgecolor='black')
plt.title('Distribution of Text Lengths')
plt.xlabel('Text Length')
plt.ylabel('Frequency')
plt.show()

#===================================================================================
# Word Frequency Plot
from collections import Counter
words_counter = Counter(' '.join(df['cleaned_text']).split())
common_words = words_counter.most_common(20)
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.bar([word[0] for word in common_words], [count[1] for count in common_words], color='orange')
plt.title('Top 20 Most Common Words')
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.xticks(rotation=45, ha='right')
plt.show()

#===================================================================================
# Sentiment Distribution Over Time:
# Assuming 'vader_sentiment' is the column containing sentiment labels
sentiment_distribution = df['vader_sentiment'].value_counts()

# Plotting the distribution
sentiment_distribution.plot(kind='bar', color=['green', 'red', 'blue'])
plt.title('Distribution of Sentiment Labels')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()

# The trained model is applied to new text ("I really enjoyed the movie. It was fantastic!") for sentiment prediction.
# The distribution of text lengths (number of words) in the dataset is visualized using a histogram.
# A bar plot is created to display the top 20 most common words in the cleaned text.
# The distribution of sentiment labels over time is visualized
#====================================================================================


