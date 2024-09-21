#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize


# In[12]:


zomato_df = pd.read_csv('zomato_reviews.csv')


# In[13]:


zomato_df['review'].fillna('', inplace=True)


# In[16]:


# Stemming using NLTK's Porter Stemmer
stemmer = PorterStemmer()
zomato_df['review'] = zomato_df['review'].apply(lambda x: ' '.join([stemmer.stem(word) for word in word_tokenize(x)]))


# In[17]:


X = zomato_df['review']
y = zomato_df['rating']


# In[18]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[19]:


# Create a pipeline with a CountVectorizer, TfidfTransformer, and Multinomial Naive Bayes Classifier
model = make_pipeline(CountVectorizer(), TfidfTransformer(), MultinomialNB())


# In[20]:


model.fit(X_train, y_train)


# In[21]:


y_pred = model.predict(X_test)


# In[22]:


accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")


# In[25]:


unique_classes = zomato_df['rating'].unique()

print("Unique Classes:", unique_classes)
print("Number of Classes:", len(unique_classes))


# In[ ]:




