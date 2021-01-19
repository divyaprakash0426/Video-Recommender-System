import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gzip
import xgboost as xg
import itertools
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from datetime import datetime
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.ensemble import ExtraTreesRegressor
from numpy import *
from scipy.sparse.linalg import svds
from numpy import linalg as la
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix

#Data can be treated as python dictionary objects.
#A simple script to read any of the above the data is as follows:
def opening(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield eval(l)

def initial_data_from_dict(path):
  i = 0
  initial_data = {}
  for d in opening(path):
    initial_data[i] = d
    i += 1
  return pd.DataFrame.from_dict(initial_data, orient='index')

def more_clean(initial_data, feature, m):
    count = initial_data[feature].value_counts()
    initial_data = initial_data[initial_data[feature].isin(count[count > m].index)]
    return initial_data

def cleaning_data(initial_data,features,m):
    fil = initial_data.asin.value_counts()
    fil2 = initial_data.reviewerID.value_counts()
    initial_data['no_of_products'] = initial_data.asin.apply(lambda x: fil[x])
    initial_data['no_of_users'] = initial_data.reviewerID.apply(lambda x: fil2[x])
    while (initial_data.asin.value_counts(ascending=True)[0]) < m or  (initial_data.reviewerID.value_counts(ascending=True)[0] < m):
        initial_data = more_clean(initial_data,features[0],m)
        initial_data = more_clean(initial_data,features[1],m)
    return initial_data

def data():
    print('loading data...')
    initial_data = initial_data_from_dict('C:/Users/divya/New folder/reviews_Amazon_Instant_Video_5.json.gz')
    initial_data['reviewTime'] =initial_data['reviewTime'].apply(lambda x: datetime.strptime(x, '%m %d, %Y'))
    initial_data['datetime'] = pd.to_datetime(initial_data.reviewTime, unit='s')

    raw_data = cleaning_data(initial_data, ['asin', 'reviewerID'], 2)
    raw_data['userid'] = pd.factorize(raw_data['reviewerID'])[0]
    raw_data['videoid'] = pd.factorize(raw_data['asin'])[0]

    sc = MinMaxScaler()

    raw_data['time']=sc.fit_transform(raw_data['reviewTime'].values.reshape(-1,1))
    raw_data['numberuser']=sc.fit_transform(raw_data['no_of_users'].values.reshape(-1,1))
    raw_data['numberprod']=sc.fit_transform(raw_data['no_of_products'].values.reshape(-1,1))
    raw_data['reviewTime'] =  pd.to_datetime(raw_data['reviewTime'], format='%Y-%b-%d:%H:%M:%S.%f')    
    raw_data['weekend']=raw_data['reviewTime'].dt.dayofweek>=5
    raw_data['weekend']=raw_data['weekend'].astype(int)

    First = raw_data.loc[:,['userid','videoid']]
    Second = raw_data.loc[:,['userid','videoid','time']]
    Third = raw_data.loc[:,['userid','videoid','time','numberuser','numberprod']]
    
    y = raw_data.overall

    # train_test split
    train_1,test_1,y_train,y_test = train_test_split(First,y,test_size=0.3,random_state=2017)
    train_2,test_2,y_train,y_test = train_test_split(Second,y,test_size=0.3,random_state=2017)
    train_3,test_3,y_train,y_test = train_test_split(Third,y,test_size=0.3,random_state=2017)
    train = np.array(train_1.join(y_train))
    test = np.array(test_1.join(y_test))
    videoid2videoid = raw_data.asin.unique()
    data_mixed = First.join(y)
    total_p = data_mixed['videoid'].unique().shape[0]
    total_u = data_mixed['userid'].unique().shape[0]
    
    # make the user-item uv_table
    uv_table = np.zeros([total_u,total_p])
    z = np.array(data_mixed)
    for line in z:
        u,p,s = line
        if uv_table[int(u)][int(p)] < int(s):
            uv_table[int(u)][int(p)] = int(s) 
    print('the uv_table\'s shape is:' )
    print(uv_table.shape)
    return z, total_u,total_p,videoid2videoid,train,test,uv_table,raw_data

z, total_u,total_p,videoid2videoid,train,test,uv_table,raw_data = data()