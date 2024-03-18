#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


cd = pd.read_csv('Titanic-Dataset.csv')


# In[3]:


cd


# In[4]:


cd.head()


# In[5]:


cd.tail()


# In[6]:


cd.iloc()


# In[7]:


cd.loc()


# In[8]:


cd.info()


# In[9]:


cd = cd.drop([2,3], axis=0)


# In[10]:


cd


# In[11]:


(len(cd.columns))
# to count number of columns only


# In[12]:


(len(cd))
# to count number of rows only


# In[13]:


cd.shape
# to find no. of rows and columns at same time


# In[14]:


cd.size
# multiplies rows and columns together


# In[15]:


cd.describe()
# to count mean  , max , min , std


# In[16]:


import matplotlib.pyplot as plt 


# In[17]:


import seaborn as sns


# In[18]:


# Create the heatmap to check missing values

sns.heatmap(cd.corr(), annot=True, fmt='.1f')
plt.title('missing values')
plt.xlabel('Columns Name')
plt.ylabel('Average of Missing Values')
plt.show()


# In[19]:


Embarked = pd.get_dummies(cd['Embarked'],drop_first=True)
# creating dummies


# In[20]:


cd.drop([ 'Embarked'],axis=1,inplace=True)
# this will help to drop columns with name sex and embarked


# In[21]:


cd


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
# to properly arrange columns


# In[23]:


cd


# In[24]:


cd = pd.concat([cd,Embarked],axis=1)


# In[25]:


cd


# In[26]:


cd['Female'] = (cd['Sex'] == 'female').astype(int)


# In[27]:


cd


# In[28]:


cd['Male'] = (cd['Sex'] == 'female').astype(int)


# In[29]:


cd


# In[30]:


sns.countplot(x = 'Survived' , data=cd)
# bar graph to show no.of survived and unsurvived  


# In[31]:


sns.countplot(x = 'Pclass' , data=cd)
# bar graph to show counting of different type of passenger class


# In[32]:


sns.countplot(x = 'Survived' , data=cd ,hue='Male')


# In[33]:


sns.countplot(x = 'Survived' , data=cd ,hue='Female')


# In[34]:


plt.figure(figsize=(10,5))
plt.hist(cd['Fare'], bins = 50)
plt.title("Fare Distribution")
plt.xlabel('Fare')
plt.ylabel('Frequency')
plt.show()


# In[35]:


cd.Ticket.value_counts()


# In[36]:


cd


# In[37]:


cd.Fare.value_counts()


# In[38]:


cd.Male.value_counts()


# In[39]:


cd.Female.value_counts()


# In[40]:


from sklearn.model_selection import train_test_split


# In[41]:


cols = ['Name', 'Ticket', 'Cabin']
cd = cd.drop(cols, axis=1)


# In[42]:


cd


# In[43]:


cd = cd.dropna()
# missing row having missing values


# In[44]:


cd


# In[45]:


cd = cd.drop(['Pclass', 'Sex'], axis=1)
# droping unwanted columns


# In[46]:


cd


# In[ ]:




