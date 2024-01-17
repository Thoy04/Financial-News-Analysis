import numpy as np
import pandas as pd
import string
import nltk
import seaborn as sns
import torch
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
from wordcloud import WordCloud,STOPWORDS
# nltk.download('all')

fin_df = pd.read_csv('C:/Users/sierr/Desktop/MSDS/Visualization & Unstructured Data Analysis/Project 4/fin_data.csv',encoding_errors='ignore')
fin_df = fin_df.rename(columns={'neutral':'sentiment','According to Gran , the company has no plans to move all production to Russia , although that is where the company is growing .':'Headline'})
fin_df.drop_duplicates(inplace=True)
fin_df.reset_index(inplace=True)


# create preprocess_text function
def preprocess_text(text):
    # Tokenize the text
    punct_exclude = set(string.punctuation)
    punct_free = ''.join([ch for ch in text if ch not in punct_exclude])
    tokens = word_tokenize(punct_free.lower())
    # Remove stop words
    filtered_tokens = [token for token in tokens if token not in stopwords.words('english')]
    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    # Join the tokens back into a string
    processed_text = ' '.join(lemmatized_tokens)
    return processed_text

# create get_sentiment function
def get_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)
    # sentiment =  if scores['pos'] > 0 else 0
    if scores['pos'] > 0:
        sentiment = 'positive'
    elif scores['pos'] == 0 and scores['neg'] > 0:
        sentiment = 'negative'
    else:
        sentiment = 'neutral'
    return sentiment


# run functions on data frame
fin_df['headline_clean'] = fin_df['Headline'].apply(preprocess_text)
fin_df['sentiment_prediction'] = fin_df['headline_clean'].apply(get_sentiment)
# check accuracy
are_equal = hf_preds['sentiment'] == hf_preds['sentiment_prediction']
len(hf_preds[are_equal])/len(hf_preds)


def get_model(model_name):
    model_name = model_name
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    return classifier


model = get_model("oferweintraub/bert-base-finance-sentiment-noisy-search")
model(fin_df['Headline'][0])


#def get_labels()
labels = []
for i in range(4200,4840):
    ans = model(fin_df['Headline'][i])
    labels.append(ans[0]['label'])
#    return labels


#labels = get_labels()
fin_df['sentiment_prediction'] = labels
hf_preds = fin_df


positives = fin_df[fin_df['sentiment'] == 'positive']
negatives = fin_df[fin_df['sentiment'] == 'negative']
neutrals = fin_df[fin_df['sentiment'] == 'neutral']


def create_wordcloud(df):
    wordcloud= WordCloud(width=1000, height=500, stopwords=STOPWORDS, background_color='white').generate(''.join(df['Headline']))
    plt.figure(figsize=(15,8))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()


create_wordcloud(positives)
create_wordcloud(negatives)

true_labels = fin_df['sentiment']
predicted_labels = fin_df['sentiment_prediction']


def make_matrix(trues, predictions):
    cm = confusion_matrix(trues, predictions, labels=['positive', 'neutral', 'negative'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['positive', 'neutral', 'negative'], yticklabels=['positive', 'neutral', 'negative'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()


make_matrix(true_labels, predicted_labels)