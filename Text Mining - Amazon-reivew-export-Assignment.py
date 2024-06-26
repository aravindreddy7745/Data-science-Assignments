# Importing the data
import pandas as pd
df=pd.read_excel("Amazon-reivew-export.xlsx")
df.info()
df.head()
df.shape

import pandas as pd
stop_words=pd.read_csv("stop.txt")
stop_words


import pandas as pd
pos_words=pd.read_csv("positive-words.txt",sep='\t')
pos_words


import pandas as pd
neg_words=pd.read_csv("negative-words.txt",sep="\t",encoding='latin-1')
neg_words

#==================================================================================
# Emotional analysis
# cleaning the tweets
one_tweet = df.iloc[4]['Review Content']
one_tweet

book = df.iloc[:,8:9]
book

stop_words_set = set(stop_words['a'])
# Convert DataFrame column to a set
#===================================================================================
# Emotion Mining
def get_emotion(review):
    review_words = set()  # Create an empty set for each review
    words = review.split()  # Split the review into words

    # Remove stop words from the words list
    words = [word for word in words if word.lower() not in stop_words_set]

    review_words.update(words)  # Add the words to the set

    positive_matches = set(review_words).intersection(pos_words)
    negative_matches = set(review_words).intersection(neg_words)

    positive_count = len(positive_matches)
    negative_count = len(negative_matches)

    # Determine the emotion label
    if positive_count > negative_count:
        emotion_label = "Positive"
    elif negative_count > positive_count:
        emotion_label = "Negative"
    else:
        emotion_label = "Neutral"

    # Calculate the emotion level
    emotion_level = positive_count - negative_count

    return emotion_label, emotion_level

df.get("Review Content")

df.get("Review Title")

#======================================================================================
# Apply the function to create new 'emotion_label' and 'emotion_level' columns
df[['emotion_label', 'emotion_level']] = df['helpful'].apply(get_emotion).apply(pd.Series)
df

# Display emotion counts and pivot table
positive_count = df['emotion_label'].value_counts().get('Positive', 0)
negative_count = df['emotion_label'].value_counts().get('Negative', 0)
Neutral_count = df['emotion_label'].value_counts().get('Neutral', 0)

print("Positive Emotion Count:", positive_count)
print("Negative Emotion Count:", negative_count)
print("Neutral Emotion Count:", Neutral_count)

emotion_pivot_table  = book.pivot_table(index=['emotion_label'],aggfunc={"emotion_label":'count'})
print(emotion_pivot_table)

# Plot the bar graph
import matplotlib.pyplot as plt
emotion_counts = book['emotion_label'].value_counts()
plt.bar(emotion_counts.index, emotion_counts.values)
plt.xlabel('Emotion Label')
plt.ylabel('Count')
plt.title('Emotion Label Count')
plt.show()

#======================================================================================
"""## **Text Preprocessing**"""

df = book.iloc[:,0:1]
df
# Text Preprocessing
df1 = [x.strip() for x in df.helpful] # remove both the leading and the trailing characters
df1 = [x for x in df1 if x] # removes empty strings, because they are considered in Python as False
df1[0:10]

# Joining the list into one string/text
text = ' '.join(df1)
text

# Punctuation removal
import string
no_punc_text = text.translate(str.maketrans('', '', string.punctuation)) #with arguments (x, y, z) where 'x' and 'y'
# must be equal-length strings and characters in 'x'
# are replaced by characters in 'y'. 'z'
# is a string (string.punctuation here)
no_punc_text

#==================================================================================
import nltk
nltk.download('punkt')

#==================================================================================
#Tokenization
from nltk.tokenize import word_tokenize
text_tokens = word_tokenize(no_punc_text)
print(text_tokens[0:50])
len(text_tokens)

#==================================================================================
# Lowercasing
lower_words = [x.lower() for x in text_tokens]
print(lower_words[0:25])

#====================================================================================
#Stemming
from nltk.stem import PorterStemmer
ps = PorterStemmer()
stemmed_tokens = [ps.stem(word) for word in lower_words]
print(stemmed_tokens[0:40])
len(stemmed_tokens)

#===================================================================================
# Lemmatization
import spacy
# NLP english language model of spacy library
nlp = spacy.load('en_core_web_sm')
nlp

# lemmas being one of them, but mostly POS, which will follow later
doc = nlp(' '.join(stemmed_tokens))
print(doc[0:40])
lemmas = [token.lemma_ for token in doc]
print(lemmas[0:25])
len(lemmas)

#=================================================================================
# Feature Extraction using CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(lemmas)

print(vectorizer.get_feature_names_out()[50:100])
print(X.toarray()[50:100])

print(X.toarray().shape)

# bigram and trigram
vectorizer_ngram_range=CountVectorizer(analyzer='word',ngram_range=(1,3),max_features=(100))
bow_matrix_ngram = vectorizer_ngram_range.fit_transform(book["helpful"])
bow_matrix_ngram
print(vectorizer_ngram_range.get_feature_names_out())
print(bow_matrix_ngram.toarray())

# N- Grams
def generate_ngrams(text, n):
    words = text.split()
    ngrams = []
    for i in range(len(words) - n + 1):
        ngram = " ".join(words[i:i+n])
        ngrams.append(ngram)
    return ngrams
text_ngrams = generate_ngrams(text, 2)
print("Bi-grams:", text_ngrams)

#==================================================================================
# Example usage
text = "Naive Bayes is considered one of the most effective data mining algorithms. It is a simple probabilistic algorithm for the classification tasks."
n = 3  # You can change 'n' to generate different n-grams (e.g., bigrams, trigrams, etc.)

result = generate_ngrams(text, n)
print(f"{n}-grams: {result}")

def generate_ngrams_from_tokens(tokens, n):
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i+n])
        ngrams.append(ngram)
    return ngrams

#==================================================================================
# Example usage
tokens = ['Naive Bayes is', 'Bayes is considered', 'is considered one', 'considered one of', 'one of the', 'of the most', 'the most effective', 'most effective data', 'effective data mining', 'data mining algorithms.', 'mining algorithms. It', 'algorithms. It is', 'It is a', 'is a simple', 'a simple probabilistic', 'simple probabilistic algorithm', 'probabilistic algorithm for', 'algorithm for the', 'for the classification', 'the classification tasks']
n = 2  # Generating bigrams

result = generate_ngrams_from_tokens(tokens, n)
print(f"{n}-grams: {result}")

#Bi-grams
def generate_bigrams(text):
    words = text.split()
    bigrams = [(words[i], words[i + 1]) for i in range(len(words) - 1)]
    return bigrams

#==================================================================================
# Example usage
text = "Naive Bayes is considered one of the most effective data mining algorithms. It is a simple probabilistic algorithm for the classification tasks."
print("Bi-grams:", result)

def generate_bigrams_from_tokens(tokens):
    bigrams = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
    return bigrams

#==================================================================================
# Example usage
tokens =  ['Naive Bayes is', 'Bayes is considered', 'is considered one', 'considered one of', 'one of the', 'of the most', 'the most effective', 'most effective data', 'effective data mining', 'data mining algorithms.', 'mining algorithms. It', 'algorithms. It is', 'It is a', 'is a simple', 'a simple probabilistic', 'simple probabilistic algorithm', 'probabilistic algorithm for', 'algorithm for the', 'for the classification', 'the classification tasks']
result = generate_bigrams_from_tokens(tokens)
print("Bi-grams:", result)

#====================================================================================
# TFidf vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer_n_gram_max_features = TfidfVectorizer(norm="l2",analyzer='word', ngram_range=(1,3), max_features = 500)
tf_idf_matrix_n_gram_max_features =vectorizer_n_gram_max_features.fit_transform(book)
print(vectorizer_n_gram_max_features.get_feature_names_out())
print(tf_idf_matrix_n_gram_max_features.toarray())

# Commented out IPython magic to ensure Python compatibility.
# Generate wordcloud
# Import packages
import matplotlib.pyplot as plt
# %matplotlib inline
from wordcloud import WordCloud, STOPWORDS
# Define a function to plot word cloud
def plot_cloud(wordcloud):
    # Set figure size
    plt.figure(figsize=(40, 30))
    # Display image
    plt.imshow(wordcloud)
    # No axis details
    plt.axis("off");

# =====================================================================================
# Generate wordcloud
stopwords = set(STOPWORDS)
stopwords.add('will')
wordcloud = WordCloud(width = 3000, height = 2000, background_color='black', max_words=100,colormap='Set2',stopwords=stopwords).generate(text)
# Plot WordCloud
plt.figure(figsize=(40, 30))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()
# The code aims to analyze the emotions expressed in product reviews, classifying them as positive, negative, or neutral based on the presence of positive and negative words.
# Text preprocessing techniques such as tokenization, stemming, and lemmatization are applied to prepare the text data for analysis.
# Feature extraction methods, including CountVectorizer and TF-IDF, are employed to represent the text data numerically.
# The word cloud provides a visual representation of important words in the reviews.
#=======================================================================================
