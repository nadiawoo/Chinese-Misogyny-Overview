#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.metrics import precision_score, recall_score, f1_score
from scipy.stats import mode
import re


# In[2]:


##SWSR dataset
comments = pd.read_csv('SWSR_SexComment.csv')
comments.head()


# In[104]:


df = comments.copy()


# In[4]:


df["tokens"] = df['comment_text'].astype(str).apply(lambda x: list(jieba.cut(x)))

#training word2vec
w2v_model = Word2Vec(sentences=df["tokens"], vector_size=100, window=5, min_count=2, workers=4, seed=42)

def get_avg_vector(tokens, model):
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

df["vector"] = df["tokens"].apply(lambda x: get_avg_vector(x, w2v_model))
comment_vectors = np.vstack(df["vector"].to_numpy())


# In[5]:


kmeans = KMeans(n_clusters=2, random_state=42)
df["cluster"] = kmeans.fit_predict(comment_vectors)


# In[6]:


#evaluating using the pre-labeled data
cluster_map = {}
true_labels = df["label"]

for cluster in df["cluster"].unique():
    majority_label = mode(true_labels[df["cluster"] == cluster], keepdims=True).mode[0]
    cluster_map[cluster] = majority_label
    
df["aligned_pred"] = df["cluster"].map(cluster_map)


precision = precision_score(true_labels, df["aligned_pred"])
recall = recall_score(true_labels, df["aligned_pred"])
f1 = f1_score(true_labels, df["aligned_pred"])

print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")


# In[7]:


df.groupby("cluster")["label"].value_counts(normalize=True)
#this comparison shows that regardless of the cluster, the two categories hold the same weight


# In[8]:


pd.set_option('display.max_colwidth', None)
for c in df["cluster"].unique():
    print(f"\nCluster {c} sample:")
    display(df[df["cluster"] == c][["comment_text", "label"]].sample(15, random_state=42))


# In[9]:


#it seems like there are some ambiguities in the labelling made by the authors, where some sexist posts are misclassified or overlooked. this may have potentially contributed to the lack of precision & accuracy?

#if we were to look at each clusters, there seems to be some commonalities such as cluster 1 has shorter, more harsh toned language, whereas cluster 0 has longer and more thought/policy oriented topics. 


# In[10]:


df['aligned_pred'].unique() #it seems that this pre-liminary method onnly predicts 0 which is non-misogynistic


# In[11]:


#if the labelling was the issue for evaluation, lets try doing the same thing but for a manually labeled dataset which is shorter but hopefully more consistent.


# In[12]:


alt = pd.read_csv("sampled_keyword_df.csv")


# In[13]:


alt.head()


# In[14]:


def clean_text(text):
    if not text:
        return ""
    # Remove hyperlinks and anchor tags
    text = re.sub(r"http\S+|www\S+|<a.*?>|</a>", "", text)
    # Remove all span classes (surl-text, url-icon, and others)
    text = re.sub(r"<span.*?>.*?</span>", "", text)
    # Remove hashtags (content inside ##)
    text = re.sub(r"#.*?#", "", text)
    # Remove any remaining HTML tags
    text = re.sub(r"<.*?>", "", text)
    # Remove emojis and special characters
    text = re.sub(r"[^\w\s\u4e00-\u9fff]", "", text)
    return text.strip()

# Apply the function to the text column
alt["cleaned_text"] = alt["text"].apply(clean_text)

# Display cleaned text
alt[['cleaned_text']].head()


# In[40]:


#because we are only doing a binary labelling, i will make the -1 (feminist) into 0s
alt["misogyny_label"] = alt["misogyny_label"].replace(-1, 0) 
alt = alt.dropna()


# In[140]:


alt["tokens"] = alt['cleaned_text'].astype(str).apply(lambda x: list(jieba.cut(x)))

w2v_model = Word2Vec(sentences=alt["tokens"], vector_size=100, window=5, min_count=2, workers=4, seed=42)

def get_avg_vector(tokens, model):
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

alt["vector"] = alt["tokens"].apply(lambda x: get_avg_vector(x, w2v_model))
comment_vectors = np.vstack(alt["vector"].to_numpy())

kmeans = KMeans(n_clusters=2, random_state=42)
alt["cluster"] = kmeans.fit_predict(comment_vectors)

cluster_map = {}
true_labels = alt["misogyny_label"]

for cluster in alt["cluster"].unique():
    majority_label = mode(true_labels[alt["cluster"] == cluster], keepdims=True).mode[0]
    cluster_map[cluster] = majority_label

alt["aligned_pred"] = alt["cluster"].map(cluster_map)

precision = precision_score(true_labels, alt["aligned_pred"])
recall = recall_score(true_labels, alt["aligned_pred"])
f1 = f1_score(true_labels, alt["aligned_pred"])

print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")


# In[144]:


alt.groupby("cluster")["misogyny_label"].value_counts(normalize=True) #similar skewness as the previous dataset, which leads me to believe that may the clustering is capturing something? 


# In[50]:


pd.set_option('display.max_colwidth', None)
for c in alt["cluster"].unique():
    print(f"\nCluster {c} sample:")
    display(alt[alt["cluster"] == c][["cleaned_text", "misogyny_label"]].sample(15, random_state=42))


# In[52]:


# it seems that despite there are misclassification occuring because nuances and sarcasm is not captured well, there is still an interesting pattern that distinguishes cluster 1 and 0. In both situations, one cluster is more in length than another, more emotional and more anger (often categorized as hate). So while this cannot be used to pinpoint the shifts of language used for misogyny, it does capture patterns of discussion. 


# In[122]:


df["tokens"] = df['comment_text'].astype(str).apply(lambda x: list(jieba.cut(x)))

w2v_model = Word2Vec(sentences=df["tokens"], vector_size=100, window=5, min_count=2, workers=4, seed=42)

def get_avg_vector(tokens, model):
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

df["vector"] = df["tokens"].apply(lambda x: get_avg_vector(x, w2v_model))
comment_vectors = np.vstack(df["vector"].to_numpy())


# In[124]:


inertias = []
k_range = range(2, 11)  # You can adjust the upper bound

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(comment_vectors)
    inertias.append(kmeans.inertia_)


# In[130]:


inertias #seems like 7 clusters is a decent estimate as the change between each intertia is less


# In[132]:


kmeans = KMeans(n_clusters=7, random_state=42)
df["cluster"] = kmeans.fit_predict(comment_vectors)

true_labels = df["label"]
cluster_map = {}
for cluster in df["cluster"].unique():
    majority_label = mode(true_labels[df["cluster"] == cluster], keepdims=True).mode[0]
    cluster_map[cluster] = majority_label

df["aligned_pred"] = df["cluster"].map(cluster_map)

precision = precision_score(true_labels, df["aligned_pred"])
recall = recall_score(true_labels, df["aligned_pred"])
f1 = f1_score(true_labels, df["aligned_pred"])

print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")


# In[136]:


df.groupby("cluster")["label"].value_counts(normalize=True)


# In[134]:


pd.set_option('display.max_colwidth', None)
for c in df["cluster"].unique():
    print(f"\nCluster {c} sample:")
    display(df[df["cluster"] == c][["comment_text", "label"]].sample(5, random_state=42))

