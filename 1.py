import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

from sklearn.datasets import make_multilabel_classification
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelBinarizer
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA

classif = OneVsRestClassifier(SVC(kernel='linear'))


clf = SVC()


plt.style.use('ggplot')

with open('train.json') as f:
	data=pd.read_json(f)
#print(data.columns())

#for item in data.itertuples():print(item)

data['interest_level']=data['interest_level'].map({'high':2,'medium':1,'low':0})

k=data.features

k=data.features
aa=[];
for item in k:
	aa.append(len(item))

data.insert(1, 'feature_len',aa)

del data['photos']
del data['created']
del data['description']
del data['display_address']
del data['features']
del data['listing_id']
del data['street_address']
del data['building_id']
del data['manager_id']
del data['latitude']
del data['longitude']

y=data['interest_level'];
del data['interest_level']

data1=data.as_matrix()
y1=y.as_matrix()

a=set();
for item in k:
	for i in item:
		a.add(i)

clf.fit(data1[1:20000], y1[1:20000])
classif.fit(data1[1:2000], y1[1:2000])
accuracy_score(y1,clf.predict(data1))
accuracy_score(y1,classif.predict(data1))



#data['manager_id'].value_counts().min()
#data['manager_id'].value_counts(normalize=False, sort=True, ascending=False)

#write the rfesultinto file, and read it back
#data.to_json(path_or_buf="tt")
#with open('tt') as f: data1=pd.read_json(f)



#manager_id'
#plt.figure();
#data['manager_id'].plot.hist()
#data.plot(kind='bar')
#for i in range(0,len(data.index)):
#	print(data.iloc[i].interest_level)
#for item in data.itertuples():print(item)
#for key,vaue in data.items():
#	print(data[data.building_id])
#	print(key)
def df_to_sarray(df):
    """
    Convert a pandas DataFrame object to a numpy structured array.
    This is functionally equivalent to but more efficient than
    np.array(df.to_array())

    :param df: the data frame to convert
    :return: a numpy structured array representation of df
    """

    v = df.values
    cols = df.columns
    types = [(cols[i].encode(), df[k].dtype.type) for (i, k) in enumerate(cols)]
    dtype = np.dtype(types)
    z = np.zeros(v.shape[0], dtype)
    for (i, k) in enumerate(z.dtype.names):
        z[k] = v[:, i]
    return z
