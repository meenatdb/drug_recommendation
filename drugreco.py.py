
# coding: utf-8

# In[4]:


import pandas as pd #Analysis 
import numpy as np #Analysis 
from scipy.stats import norm #Analysis 
from sklearn.preprocessing import StandardScaler #Analysis 
from scipy import stats #Analysis 
import warnings 
warnings.filterwarnings('ignore')

import gc

import os
import string

from sklearn import model_selection, preprocessing, metrics, ensemble, naive_bayes, linear_model
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
import lightgbm as lgb


# In[5]:


df_train = pd.read_csv("C:/Users/dtallapr/OneDrive - Capgemini/Desktop/Analysis/proactive pitches/TTR/drugsComTest_raw.csv", parse_dates=["date"])
df_test = pd.read_csv("C:/Users/dtallapr/OneDrive - Capgemini/Desktop/Analysis/proactive pitches/TTR/drugsComTrain_raw.csv", parse_dates=["date"])


# In[6]:


df_all = pd.concat([df_train,df_test])


# In[7]:


condition_dn = df_all.groupby(['condition'])['drugName'].nunique().sort_values(ascending=False)


# In[8]:


df_all[df_all['condition']=='3</span> users found this comment helpful.']


# In[9]:


condition_dn = df_all.groupby(['condition'])['drugName'].nunique().sort_values(ascending=False)


# In[10]:


percent = (df_all.isnull().sum()).sort_values(ascending=False)


# In[11]:


print("Missing value (%):", 1200/df_all.shape[0] *100)


# In[12]:


df_train = df_train.dropna(axis=0)
df_test = df_test.dropna(axis=0)


# In[13]:


df_all = pd.concat([df_train,df_test]).reset_index()
del df_all['index']
percent = (df_all.isnull().sum()).sort_values(ascending=False)


# In[14]:


all_list = set(df_all.index)
span_list = []
for i,j in enumerate(df_all['condition']):
    if '</span>' in j:
        span_list.append(i)


# In[15]:


new_idx = all_list.difference(set(span_list))
df_all = df_all.iloc[list(new_idx)].reset_index()
del df_all['index']


# In[17]:


df_condition = df_all.groupby(['condition'])['drugName'].nunique().sort_values(ascending=False)
df_condition = pd.DataFrame(df_condition).reset_index()


# In[18]:


df_condition_1 = df_condition[df_condition['drugName']==1].reset_index()


# In[19]:


all_list = set(df_all.index)
condition_list = []
for i,j in enumerate(df_all['condition']):
    for c in list(df_condition_1['condition']):
        if j == c:
            condition_list.append(i)
            
new_idx = all_list.difference(set(condition_list))
df_all = df_all.iloc[list(new_idx)].reset_index()
del df_all['index']


# In[20]:


from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer


# In[23]:


stops = set(stopwords.words('english'))
stopwords = set(stopwords.words('english'))
more_stopwords = {'one', 'br', 'Po', 'th', 'sayi', 'fo', 'Unknown'}
stopwords = stopwords.union(more_stopwords)
not_stop = ["aren't","couldn't","didn't","doesn't","don't","hadn't","hasn't","haven't","isn't","mightn't","mustn't","needn't","no","nor","not","shan't","shouldn't","wasn't","weren't","wouldn't"]
for i in not_stop:
    stops.remove(i)


# In[24]:


from sklearn import model_selection, preprocessing, metrics, ensemble, naive_bayes, linear_model
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
import lightgbm as lgb

pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999
from bs4 import BeautifulSoup
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split
from sklearn import metrics

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers


# In[25]:


stemmer = SnowballStemmer('english')

def review_to_words(raw_review):
    # 1. Delete HTML 
    review_text = BeautifulSoup(raw_review, 'html.parser').get_text()
    # 2. Make a space
    letters_only = re.sub('[^a-zA-Z]', ' ', review_text)
    # 3. lower letters
    words = letters_only.lower().split()
    # 5. Stopwords 
    meaningful_words = [w for w in words if not w in stops]
    # 6. Stemming
    stemming_words = [stemmer.stem(w) for w in meaningful_words]
    # 7. space join words
    return( ' '.join(stemming_words))


# In[26]:


df_all['review_clean'] = df_all['review'].apply(review_to_words)


# In[27]:


# Make a rating
df_all['sentiment'] = df_all["rating"].apply(lambda x: 1 if x > 5 else 0)


# In[28]:


df_train, df_test = train_test_split(df_all, test_size=0.33, random_state=42) 


# In[29]:



from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

vectorizer = CountVectorizer(analyzer = 'word', 
                             tokenizer = None,
                             preprocessor = None, 
                             stop_words = None, 
                             min_df = 2,
                             ngram_range=(4, 4),
                             max_features=1000
                            )
vectorizer


# In[30]:



pipeline = Pipeline([
    ('vect', vectorizer),
])


# In[31]:


train_data_features = pipeline.fit_transform(df_train['review_clean']).todense()
test_data_features = pipeline.fit_transform(df_test['review_clean']).todense()


# In[32]:


from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Bidirectional, LSTM, BatchNormalization, Dropout
from tensorflow.python.keras.preprocessing.sequence import pad_sequences


# In[33]:



# 0. Package
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
import random

# 1. Dataset
y_train = df_train['sentiment']
y_test = df_test['sentiment']
solution = y_test.copy()

# 2. Model Structure
model = keras.models.Sequential()

model.add(keras.layers.Dense(200, input_shape=(1000,)))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.Dropout(0.5))

model.add(keras.layers.Dense(300))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.Dropout(0.5))

model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))

# 3. Model compile
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[34]:


model.summary()


# In[36]:


# 4. Train model
hist = model.fit(train_data_features, y_train, epochs=10, batch_size=64)



# 6. Evaluation
loss_and_metrics = model.evaluate(test_data_features, y_test, batch_size=32)
print('loss_and_metrics : ' + str(loss_and_metrics))


# In[37]:


sub_preds_deep = model.predict(test_data_features,batch_size=32)


# In[38]:


from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, average_precision_score
from sklearn.model_selection import KFold
from lightgbm import LGBMClassifier
from sklearn.metrics import confusion_matrix

#folds = KFold(n_splits=5, shuffle=True, random_state=546789)
target = df_train['sentiment']
feats = ['usefulCount']

sub_preds = np.zeros(df_test.shape[0])

trn_x, val_x, trn_y, val_y = train_test_split(df_train[feats], target, test_size=0.2, random_state=42) 
feature_importance_df = pd.DataFrame() 
    
clf = LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.05,
        num_leaves=30,
        #colsample_bytree=.9,
        subsample=.9,
        max_depth=7,
        reg_alpha=.1,
        reg_lambda=.1,
        min_split_gain=.01,
        min_child_weight=2,
        silent=-1,
        verbose=-1,
        )
        
clf.fit(trn_x, trn_y, 
        eval_set= [(trn_x, trn_y), (val_x, val_y)], 
        verbose=100, early_stopping_rounds=100  #30
    )

sub_preds = clf.predict(df_test[feats])
        
fold_importance_df = pd.DataFrame()
fold_importance_df["feature"] = feats
fold_importance_df["importance"] = clf.feature_importances_
feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)


# In[39]:


solution = df_test['sentiment']
confusion_matrix(y_pred=sub_preds, y_true=solution)


# In[40]:


len_train = df_train.shape[0]
df_all = pd.concat([df_train,df_test])
del df_train, df_test;
gc.collect()


# In[41]:


df_all['date'] = pd.to_datetime(df_all['date'])
df_all['day'] = df_all['date'].dt.day
df_all['year'] = df_all['date'].dt.year
df_all['month'] = df_all['date'].dt.month


# In[42]:


from textblob import TextBlob
from tqdm import tqdm
reviews = df_all['review_clean']

Predict_Sentiment = []
for review in tqdm(reviews):
    blob = TextBlob(review)
    Predict_Sentiment += [blob.sentiment.polarity]
df_all["Predict_Sentiment"] = Predict_Sentiment
df_all.head()


# In[43]:


np.corrcoef(df_all["Predict_Sentiment"], df_all["rating"])


# In[44]:


np.corrcoef(df_all["Predict_Sentiment"], df_all["sentiment"])


# In[45]:


reviews = df_all['review']

Predict_Sentiment = []
for review in tqdm(reviews):
    blob = TextBlob(review)
    Predict_Sentiment += [blob.sentiment.polarity]
df_all["Predict_Sentiment2"] = Predict_Sentiment


# In[46]:


np.corrcoef(df_all["Predict_Sentiment2"], df_all["rating"])


# In[47]:


np.corrcoef(df_all["Predict_Sentiment2"], df_all["sentiment"])


# In[48]:



df_all['count_sent']=df_all["review"].apply(lambda x: len(re.findall("\n",str(x)))+1)

df_all['count_word']=df_all["review_clean"].apply(lambda x: len(str(x).split()))

df_all['count_unique_word']=df_all["review_clean"].apply(lambda x: len(set(str(x).split())))

df_all['count_letters']=df_all["review_clean"].apply(lambda x: len(str(x)))

df_all["count_punctuations"] = df_all["review"].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))


df_all["count_words_upper"] = df_all["review"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))

df_all["count_words_title"] = df_all["review"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))

df_all["count_stopwords"] = df_all["review"].apply(lambda x: len([w for w in str(x).lower().split() if w in stops]))


df_all["mean_word_len"] = df_all["review_clean"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))


# In[49]:


df_all['season'] = df_all["month"].apply(lambda x: 1 if ((x>2) & (x<6)) else(2 if (x>5) & (x<9) else (3 if (x>8) & (x<12) else 4)))


# In[50]:


df_train = df_all[:len_train]
df_test = df_all[len_train:]


# In[51]:


from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, average_precision_score
from sklearn.model_selection import KFold
from lightgbm import LGBMClassifier

#folds = KFold(n_splits=5, shuffle=True, random_state=546789)
target = df_train['sentiment']
feats = ['usefulCount','day','year','month','Predict_Sentiment','Predict_Sentiment2', 'count_sent',
 'count_word', 'count_unique_word', 'count_letters', 'count_punctuations',
 'count_words_upper', 'count_words_title', 'count_stopwords', 'mean_word_len', 'season']

sub_preds = np.zeros(df_test.shape[0])

trn_x, val_x, trn_y, val_y = train_test_split(df_train[feats], target, test_size=0.2, random_state=42) 
feature_importance_df = pd.DataFrame() 
    
clf = LGBMClassifier(
        n_estimators=10000,
        learning_rate=0.10,
        num_leaves=30,
        #colsample_bytree=.9,
        subsample=.9,
        max_depth=7,
        reg_alpha=.1,
        reg_lambda=.1,
        min_split_gain=.01,
        min_child_weight=2,
        silent=-1,
        verbose=-1,
        )
        
clf.fit(trn_x, trn_y, 
        eval_set= [(trn_x, trn_y), (val_x, val_y)], 
        verbose=100, early_stopping_rounds=100  #30
    )

sub_preds = clf.predict(df_test[feats])
        
fold_importance_df = pd.DataFrame()
fold_importance_df["feature"] = feats
fold_importance_df["importance"] = clf.feature_importances_
feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)


# In[52]:


confusion_matrix(y_pred=sub_preds, y_true=solution)


# In[53]:


cols = feature_importance_df[["feature", "importance"]].groupby("feature").mean().sort_values(
    by="importance", ascending=False)[:50].index

best_features = feature_importance_df.loc[feature_importance_df.feature.isin(cols)]

# plt.figure(figsize=(14,10))
# sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
# plt.title('LightGBM Features (avg over folds)')
# plt.tight_layout()
# plt.savefig('lgbm_importances.png')


# In[54]:


# import dictionary data
word_table = pd.read_csv("C:/Users/dtallapr/OneDrive - Capgemini/Desktop/Analysis/proactive pitches/TTR/inquirerbasic.csv")


# In[56]:


##1. make list of sentiment
#Positiv word list   
temp_Positiv = []
Positiv_word_list = []
for i in range(0,len(word_table.Positiv)):
    if word_table.iloc[i,2] == "Positiv":
        temp = word_table.iloc[i,0].lower()
        temp1 = re.sub('\d+', '', temp)
        temp2 = re.sub('#', '', temp1) 
        temp_Positiv.append(temp2)

Positiv_word_list = list(set(temp_Positiv))
len(temp_Positiv)
len(Positiv_word_list)  #del temp_Positiv

#Negativ word list          
temp_Negativ = []
Negativ_word_list = []
for i in range(0,len(word_table.Negativ)):
    if word_table.iloc[i,3] == "Negativ":
        temp = word_table.iloc[i,0].lower()
        temp1 = re.sub('\d+', '', temp)
        temp2 = re.sub('#', '', temp1) 
        temp_Negativ.append(temp2)

Negativ_word_list = list(set(temp_Negativ))
len(temp_Negativ)
len(Negativ_word_list)  #del temp_Negativ


# In[57]:


##2. counting the word 98590
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(vocabulary = Positiv_word_list)
content = df_test['review_clean']
X = vectorizer.fit_transform(content)
f = X.toarray()
f = pd.DataFrame(f)
f.columns=Positiv_word_list
df_test["num_Positiv_word"] = f.sum(axis=1)

vectorizer2 = CountVectorizer(vocabulary = Negativ_word_list)
content = df_test['review_clean']
X2 = vectorizer2.fit_transform(content)
f2 = X2.toarray()
f2 = pd.DataFrame(f2)
f2.columns=Negativ_word_list
df_test["num_Negativ_word"] = f2.sum(axis=1)


# In[59]:


##3. decide sentiment
df_test["Positiv_ratio"] = df_test["num_Positiv_word"]/(df_test["num_Positiv_word"]+df_test["num_Negativ_word"])
df_test["sentiment_by_dic"] = df_test["Positiv_ratio"].apply(lambda x: 1 if (x>=0.5) else (0 if (x<0.5) else 0.5))


# In[60]:


def userful_count(data):
    grouped = data.groupby(['condition']).size().reset_index(name='user_size')
    data = pd.merge(data,grouped,on='condition',how='left')
    return data
#___________________________________________________________
df_test =  userful_count(df_test) 
df_test['usefulCount'] = df_test['usefulCount']/df_test['user_size']


# In[62]:


df_test['deep_pred'] = sub_preds_deep
df_test['machine_pred'] = sub_preds

df_test['total_pred'] = (df_test['deep_pred']+ df_test['machine_pred'] + df_test['sentiment_by_dic'])*df_test['usefulCount']


# In[63]:


df_test = df_test.groupby(['condition','drugName']).agg({'total_pred' : ['mean']})
# df_test1= df_test.droplevel(0)
# df_test1.head()
# df_test2 = df_test1.droplevel(0)
# print(df_test2.columns)
# print(df_test.columns)
# print(df_test.head())
# df_test1= df_test.rename(columns= {'Unnamed: 0':'condition','Unnamed: 1':'drug','total_pred':'score'}, inplace = True)
# df_test1 = df_test.iloc[2:, :]

# In[64]:

# print(df_test1.head(10))
df_test.to_csv('C:/Users/dtallapr/OneDrive - Capgemini/Desktop/Analysis/proactive pitches/TTR/out.csv')
df = pd.read_csv("C:/Users/dtallapr/OneDrive - Capgemini/Desktop/Analysis/proactive pitches/TTR/out.csv")
df.rename(columns= {'Unnamed: 0':'condition','Unnamed: 1':'drug','total_pred':'score'}, inplace = True)
df_test1 = df.iloc[2: , :]
# print(df_test.head())
# print(df_test1.columns)
# print(df_test1.head())
# df_test1.to_csv('C:/Users/dtallapr/OneDrive - Capgemini/Desktop/Analysis/proactive pitches/TTR/out1.csv')
df2 = pd.read_csv("C:/Users/dtallapr/OneDrive - Capgemini/Desktop/Analysis/proactive pitches/TTR/drug2.csv")
final = pd.concat([df_test1, df2], ignore_index=True)
# final = pd.concat(
#     map(pd.read_csv, ['C:/Users/dtallapr/OneDrive - Capgemini/Desktop/Analysis/proactive pitches/TTR/out1.csv', 'C:/Users/dtallapr/OneDrive - Capgemini/Desktop/Analysis/proactive pitches/TTR/drug2.csv']), ignore_index=True)
# print(final.head())
final.to_csv('C:/Users/dtallapr/OneDrive - Capgemini/Desktop/Analysis/proactive pitches/TTR/output.csv')