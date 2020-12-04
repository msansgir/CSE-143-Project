#!/usr/bin/env python
# coding: utf-8

# In[1]:




# In[2]:


import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from pandas.io.json import json_normalize


# In[3]:





# In[4]:


import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score,confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from pandas import DataFrame


# In[5]:


scope = ['https://spreadsheets.google.com/feeds']
credentials = ServiceAccountCredentials.from_json_keyfile_name('./mrtutoring-1ed099ef30fd.json', scope)
gc = gspread.authorize(credentials)


# In[6]:


spreadsheet_key = '1E4_GzXYpQDldZP0jVgAMeuS2iutICLWEdSlRvawjTpA'
book = gc.open_by_key(spreadsheet_key)
worksheet = book.worksheet("data")
table = worksheet.get_all_values()
table


# In[8]:


df = pd.DataFrame(table[1:], columns=table[0])


# In[9]:


df.columns.tolist()


# In[10]:


df.drop(['id' ] , axis= 1, inplace = True)


# In[11]:


for i, row in df.iterrows():
    if(row['diagnosis'] == 'M'):
        row['diagnosis'] = 1
    else :
        row['diagnosis'] = 0
    


# In[12]:


for i in df.columns.tolist():
    df[i] = df[i].astype(float)
df.describe()


# In[13]:


ml_data = df.drop(['perimeter_mean', 'area_mean', 
                            'radius_worst', 'perimeter_worst', 'area_worst',
                           'perimeter_se', 'area_se', 'texture_worst',
                           'concave points_worst', 'concavity_mean', 'compactness_worst'], axis=1)
data = ml_data.drop(['diagnosis'] , axis = 1)
labels = ml_data['diagnosis']


# In[14]:


data_train, data_test, labels_train, labels_test = train_test_split(data,labels, test_size=0.3, random_state=123)


# In[15]:


norm = MinMaxScaler().fit(data_train)
std = StandardScaler().fit(data_train)
# transform training data
data_train_norm = norm.transform(data_train)
data_test_norm = norm.transform(data_test)

data_test_std = std.transform(data_test)
data_train_std = std.transform(data_train)


# In[16]:


import sklearn
svc = sklearn.svm.SVC()


# In[18]:


lr = LogisticRegression(max_iter = 600, n_jobs=-1, random_state=223)
knn = KNN()
dt = DecisionTreeClassifier(random_state=123)
rf = RandomForestClassifier(random_state=123)

# Define a list called classifier that contains the tuples (classifier_name, classifier)
classifiers = [('Logistic Regression', lr),
('K Nearest Neighbours', knn),
('Random Forest Classifier', rf),
('Decision Tree', dt)]              


# In[19]:


for clf_name, clf in classifiers:
    
    clf.fit(data_train_norm, labels_train)
    
    label_pred = clf.predict(data_test_norm)
   
    print('{:s} : {:.3f}'.format(clf_name, accuracy_score(labels_test, label_pred)))


# In[20]:


for clf_name, clf in classifiers:
    
    clf.fit(data_train_std, labels_train)
    
    label_pred = clf.predict(data_test_std)
   
    print('{:s} : {:.3f}'.format(clf_name, accuracy_score(labels_test, label_pred)))


# Using the best model which was the Logistic Regression Model trained with standardized data

# In[21]:


print(data_train.columns.tolist())


# In[22]:


data_for_testing = data_test[5:6]

lr.fit(data_train_std, labels_train)
prediction = clf.predict(data_for_testing)
if prediction == 0.0:
    print('Benign')
else :
    print('Malignant')


# In[23]:


import pickle


# In[25]:


pkl_filename = "pickle_model.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(lr, file)


# In[26]:


with open(pkl_filename, 'rb') as file:
    pickle_model = pickle.load(file)


# In[ ]:





# In[ ]:




