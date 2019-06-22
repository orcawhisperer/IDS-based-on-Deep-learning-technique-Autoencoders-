# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 14:19:29 2019

@author: Shyam
"""
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from numpy.random import seed
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense
from keras.models import Model
col_names2 = []

for i in range(20):
    tmp = 'feature_'+str(i)
    col_names2.append(tmp)
col_names2.append("label")

                 
train2_encoded = pd.read_csv('./train_encoded2.csv')
test2_encoded = pd.read_csv('./test_encoded2.csv')



train1_encoded = train2_encoded.iloc[:,0:20].values
test1_encoded = test2_encoded.iloc[:,0:20].values#feature extraction


train_label_encoded = train2_encoded.iloc[:,20].values
test_label_encoded = test2_encoded.iloc[:,20].values


train2_encoded.head(5)
ncol = train2_encoded.shape[1]-1

encoding_dim = 10

input_dim = Input(shape = (ncol, ))

# Encoder Layers

#encoded1 = Dense(3000, activation = 'relu')(input_dim)
#encoded2 = Dense(2750, activation = 'relu')(encoded1)
#encoded3 = Dense(2500, activation = 'relu')(encoded2)
#encoded4 = Dense(2250, activation = 'relu')(encoded3)
#encoded5 = Dense(2000, activation = 'relu')(encoded4)
#encoded6 = Dense(1750, activation = 'relu')(encoded5)
#encoded7 = Dense(1500, activation = 'relu')(encoded6)
#encoded8 = Dense(1250, activation = 'relu')(encoded7)
#encoded9 = Dense(1000, activation = 'relu')(encoded8)
#encoded10 = Dense(750, activation = 'relu')(encoded9)
#encoded11 = Dense(500, activation = 'relu')(encoded10)
#encoded12 = Dense(250, activation = 'relu')(encoded11)
#encoded13 = Dense(encoding_dim, activation = 'relu')(encoded12)
#
## Decoder Layers
#decoded1 = Dense(250, activation = 'relu')(encoded13)
#decoded2 = Dense(500, activation = 'relu')(decoded1)
#decoded3 = Dense(750, activation = 'relu')(decoded2)
#decoded4 = Dense(1000, activation = 'relu')(decoded3)
#decoded5 = Dense(1250, activation = 'relu')(decoded4)
#decoded6 = Dense(1500, activation = 'relu')(decoded5)
#decoded7 = Dense(1750, activation = 'relu')(decoded6)
#decoded8 = Dense(2000, activation = 'relu')(decoded7)
#decoded9 = Dense(2250, activation = 'relu')(decoded8)
#decoded10 = Dense(2500, activation = 'relu')(decoded9)
#decoded11 = Dense(2750, activation = 'relu')(decoded10)
#decoded12 = Dense(3000, activation = 'relu')(decoded11)
#decoded13 = Dense(ncol, activation = 'sigmoid')(decoded12)
print (encoding_dim)
"""2 Hidden Layers"""
#encoded1 = Dense(encoding_dim, activation = 'relu')(input_dim)
#decoded1 = Dense(ncol, activation = 'sigmoid')(encoded1)
"""4 Hidden Layers"""
#encoded1 = Dense(42*8, activation = 'relu')(input_dim)
#encoded2 = Dense(encoding_dim, activation = 'relu')(encoded1)
#decoded1 = Dense(42*8, activation = 'relu')(encoded2)
#decoded2 = Dense(ncol, activation = 'sigmoid')(decoded1)


"""6 Hidden Layers"""
encoded1 = Dense(42*8, activation = 'relu')(input_dim)
encoded2 = Dense(42*4, activation = 'relu')(encoded1)
encoded3 = Dense(encoding_dim, activation = 'relu')(encoded2)


decoded1 = Dense(42*8, activation = 'relu')(encoded3)
decoded2 = Dense(42*4, activation = 'relu')(decoded1)
decoded3 = Dense(ncol, activation = 'sigmoid')(decoded2)

autoencoder = Model(inputs = input_dim, outputs = decoded3)
# Combine Encoder and Deocder layers
#autoencoder = Model(inputs = input_dim, outputs = decoded2)
#autoencoder = Model(inputs = input_dim, outputs = decoded1)

# Compile the Model
autoencoder.compile(optimizer = 'sgd', loss = 'mean_squared_error',metrics=['accuracy'])
autoencoder.summary()




autoencoder.fit(x=train1_encoded, y=train1_encoded, nb_epoch = 20, batch_size = 32, shuffle = False, validation_data = (test1_encoded, test1_encoded))



#encoder = Model(inputs = input_dim, outputs = encoded2)
encoder = Model(inputs = input_dim, outputs = encoded3)
"""6 Hidden Layers"""
encoder = Model(inputs = input_dim, outputs = encoded3)
#encoder = Model(inputs = input_dim, outputs = encoded13)
encoded_input = Input(shape = (encoding_dim, ))



encoded_train = pd.DataFrame(encoder.predict(train1_encoded))
encoded_train = encoded_train.add_prefix('feature_')
encoded_train['label']=train_label_encoded



encoded_test = pd.DataFrame(encoder.predict(test1_encoded))
encoded_test = encoded_test.add_prefix('feature_')
encoded_test['label']=test_label_encoded

print(encoded_test.shape)
encoded_test.head()



encoded_train.to_csv('train_encoded3.csv', index=False)
encoded_test.to_csv('test_encoded3.csv', index=False)