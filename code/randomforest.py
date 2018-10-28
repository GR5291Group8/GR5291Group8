
# coding: utf-8

# In[6]:


import statistics
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


# In[7]:


df_vote_average = pd.read_csv("/Users/iris/Desktop/GR5291Group8-master/Data/cleaned_movie_data_vote_average.csv")


# In[8]:


Y_vote_average = df_vote_average[df_vote_average.columns[0]]
X_vote_average = df_vote_average[df_vote_average.columns[1:len(Y_vote_average)+1]]


# In[9]:


seed = 7
test_size = 0.25
X_vote_average_train, X_vote_average_test, y_vote_average_train, y_vote_average_test = train_test_split(X_vote_average, Y_vote_average, test_size=test_size, random_state=seed)


# In[10]:


model = RandomForestRegressor()
model.fit(X_vote_average_train, y_vote_average_train)


# In[11]:


y_vote_average_pred = model.predict(X_vote_average_test)


# In[12]:


statistics.mean(abs((y_vote_average_pred-y_vote_average_test)/y_vote_average_test))

