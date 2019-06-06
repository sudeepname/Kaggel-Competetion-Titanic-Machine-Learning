
# coding: utf-8

# In[1]:


'''this model is to predict the survival of an individual at the time of shipwreck as was in titanic 
name of Kaggel competetion : Titanic: Machine Learning from Disaster
here i have used 3 algorihims to predict my model and at last choose the one that produced me highes accuracy
* First i did data preprocessing 
* Then i used logistic regression if the score is above 0.5 then the model we are working is on the right path
* Final algorithm i used here is RandomForestClassifier and this model scored 0.75 in the competetion'''


# In[2]:


import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split,KFold
from sklearn.linear_model import LogisticRegression


# In[3]:


dataset_train = pd.read_csv('C:/Users/sudeep/Downloads/Titanic_ML_disaster/train.csv')
dataset_test = pd.read_csv('C:/Users/sudeep/Downloads/Titanic_ML_disaster/test.csv')


# In[4]:


dataset_train.shape


# In[5]:


dataset_test['Survived']=np.nan


# In[6]:


dataset_train['data']='train'
dataset_test['data']='test'


# In[7]:


dataset_test = dataset_test[dataset_train.columns]


# In[8]:


dataset = pd.concat([dataset_train,dataset_test],axis=0)


# In[9]:


dataset.info()


# In[10]:


dataset['Age'] = dataset['Age'].fillna(dataset['Age'].median())
dataset['Fare'] = dataset['Fare'].fillna(dataset['Fare'].median())


# In[11]:


dataset.loc[dataset["Age"] <= 6,"Age_Group"] = "Baby"
dataset.loc[(dataset["Age"] > 6) & (dataset["Age"] <= 18),"Age_Group"] = "Child"
dataset.loc[(dataset["Age"] > 18) & (dataset["Age"] <= 40),"Age_Group"] = "YoungAdult"
dataset.loc[(dataset["Age"] > 40) & (dataset["Age"] <= 60),"Age_Group"] = "Adult"
dataset.loc[dataset["Age"] > 60, "Age_Group"] = "Old"


# In[12]:


dataset["Age_Group"].value_counts()


# In[13]:


dataset['Age_Class'] = dataset['Age']*dataset['Pclass']


# In[14]:


k = dataset.columns


# In[15]:


k = dataset['Pclass'].value_counts() 


# In[16]:


k


# In[17]:


for col in k.axes[0][0:3]:
    val = 'Pclass_'+str(col)
    dataset[val]=np.where(dataset['Pclass']==col, 1,0)


# In[18]:


dataset.head()


# In[19]:


k = dataset['Embarked'].value_counts()


# In[20]:


for col in k.axes[0][0:3]:
    val = 'Embarked_'+str(col)
    dataset[val]=np.where(dataset['Embarked']==col, 1,0)


# In[21]:


dataset['Sex'] = np.where(dataset['Sex'] == 'male', 0,1)


# In[22]:


dataset['CategoricalFare'] = pd.qcut(dataset['Fare'],5)
dataset.head()


# In[23]:


def standardization(x):
    x_min = x.min()
    x_max = x.max()
    return (x-x_min)/(x_max - x_min)
    


# In[24]:


dataset['Age'] = standardization(dataset['Age'])


# In[25]:


k = dataset['Parch'].value_counts()


# In[26]:


for col in k.axes[0][0:3]:
    val = 'Parch_'+str(col)
    dataset[val]=np.where(dataset['Parch']==col,1,0)


# In[27]:


k = dataset['SibSp'].value_counts()


# In[28]:


for col in k.axes[0][0:2]:
    val = 'SibSp_'+str(col)
    dataset[val]=np.where(dataset['SibSp']==col,1,0)


# In[29]:


#Feature Engineering Starts from here
import string


# In[30]:


def substrings_in_string(big_string, substrings):
    for substring in substrings:
        if big_string.find(substring) != -1:
            return substring
    print(big_string)
    return np.nan


# In[31]:


title_list=['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev',
                    'Dr', 'Ms', 'Mlle','Col', 'Capt', 'Mme', 'Countess',
                    'Don', 'Jonkheer']


# In[32]:


dataset['Title']=dataset['Name'].map(lambda x: substrings_in_string(x, title_list))
 
#replacing all titles with mr, mrs, miss, master
def replace_titles(x):
    title=x['Title']
    if title in ['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']:
        return 'Mr'
    elif title in ['Countess', 'Mme']:
        return 'Mrs'
    elif title in ['Mlle', 'Ms']:
        return 'Miss'
    elif title =='Dr':
        if x['Sex']=='Male':
            return 'Mr'
        else:
            return 'Mrs'
    else:
        return title
dataset['Title']=dataset.apply(replace_titles, axis=1)


# In[33]:


dataset['Family_Size']=dataset['SibSp']+dataset['Parch']


# In[34]:


dataset['IsAlone']=np.where(dataset['Family_Size']>=1,1,0)


# In[35]:


dataset['Fare_Per_Person']=dataset['Fare']/(dataset['Family_Size']+1)


# In[36]:


#dataset['Fare'] = standardization(dataset['Fare'])
dataset.loc[dataset['Fare']<= 7.9 ,'Fare'] = 0
dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
dataset.loc[ dataset['Fare'] > 31, 'Fare']= 3
dataset['Fare'] = dataset['Fare'].astype(int)
dataset.head()


# In[37]:


c = pd.get_dummies(dataset['Age_Group'])


# In[38]:


dataset = pd.concat([dataset,c],axis=1)


# In[39]:


c = pd.get_dummies(dataset['Title'])


# In[40]:


dataset = pd.concat([dataset,c],axis=1)


# In[41]:


dataset.info()


# In[42]:


dataset_train = dataset[dataset['data']=='train']
dataset_test = dataset[dataset['data']=='test']
dataset_train.drop(['data'],1,inplace=True)
dataset_test.drop(['data'],1,inplace=True)


# In[43]:


dataset_train['Survived']=dataset_train['Survived'].astype(int)


# In[44]:


Y_train = dataset_train['Survived']
X_train = dataset_train.drop(['Age','Survived','PassengerId','Ticket','Cabin','Age_Group','Embarked','SibSp','Parch','Sex','Name','Pclass','Title','CategoricalFare'],1)


# In[45]:


x_train,x_test,y_train,y_test = train_test_split(X_train,Y_train,test_size = 0.2,random_state=0)


# In[46]:


clf = LogisticRegression(class_weight ='balanced')


# In[47]:


clf.fit(x_train,y_train)


# In[48]:


clf.score(x_test,y_test)


# In[49]:


from sklearn.ensemble import RandomForestClassifier 


# In[59]:


rf = RandomForestClassifier(min_samples_split=3,verbose=1,max_depth=8,n_estimators=500)


# In[60]:


from sklearn.model_selection import cross_val_score


# In[61]:


rf.fit(x_train,y_train)


# In[62]:


rf.score(x_test,y_test)


# In[63]:


print(cross_val_score(rf,X_train,Y_train,cv=10))


# In[55]:


#maximum actual score reached on this model is 85% accuraccy by random forset classifier that was obtained by performing randomsearchCV


# In[56]:


X_test =dataset_test.drop(['Age','Survived','PassengerId','Ticket','Cabin','Age_Group','Embarked','SibSp','Parch','Sex','Name','Pclass','Title','CategoricalFare'],1)


# In[57]:


prediction = rf.predict(X_test)


# In[58]:


output = pd.DataFrame({'PassengerId': dataset_test.PassengerId,'Survived': prediction})
output.to_csv('C:/Users/sudeep/Downloads/Titanic_ML_disaster/submission.csv', index=False)
output.head()

