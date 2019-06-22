# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 09:09:55 2018

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

col_names = np.array(["duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label"])



train = pd.read_csv('./KDDTrain.csv',names=col_names)

classgroup_map = {'back':'dos','buffer_overflow':'u2r','ftp_write':'r2l','guess_passwd':'r2l','imap':'r2l',
                            'ipsweep':'probe','land':'dos','loadmodule':'u2r','multihop':'r2l','neptune':'dos','nmap':'probe',
                            'perl':'u2r','phf':'r2l','pod':'dos','portsweep':'probe','rootkit':'u2r','satan':'probe',
                            'smurf':'dos','spy':'r2l','teardrop':'dos','warezclient':'r2l','warezmaster':'r2l','normal':'normal',
                            'apache2':'dos','httptunnel':'r2l','mailbomb':'dos','mscan':'probe','named':'r2l','processtable':'dos',
                            'ps':'u2r','saint':'probe','sendmail':'r2l','snmpgetattack':'r2l','snmpguess':'r2l','sqlattack':'u2r',
                            'udpstorm':'dos','worm':'dos','xlock':'r2l','xsnoop':'r2l','xterm':'u2r'}

train['label'] = train['label'].map(classgroup_map)

#print(train['labels'])  

"""
1.Label encoder should be applied only to categorical data
"""
le=LabelEncoder()
train['protocol_type'] = le.fit_transform(train['protocol_type'])
train['service'] = le.fit_transform(train['service'])
train['flag'] = le.fit_transform(train['flag'])
train['label'] = le.fit_transform(train['label'])


test = pd.read_csv('./KDDTest.csv',names=col_names)
test['label'] = test['label'].map(classgroup_map)
test['protocol_type'] = le.fit_transform(test['protocol_type'])
test['service'] = le.fit_transform(test['service'])
test['flag'] = le.fit_transform(test['flag'])
test['label'] = le.fit_transform(test['label'])



"""trainX = train.iloc[:,0:41]
trainY=train.iloc[:,41]

testX = test.iloc[:,0:41]
testY=test.iloc[:,41]"""

train.head()

train1 = train.iloc[:,0:42].values
test1 = test.iloc[:,0:42].values

train_label = train.iloc[:,41].values
test_label = test.iloc[:,41].values

train2 = train1.copy()
test2 = test1.copy()
#target = train['labels']
for i in range(0,41):
    if(max(train1[:,i])!=0):
        train1[:,i] = ((train1[:,i]-(min(train1[:,i])))/(max(train1[:,i])-(min(train1[:,i])))).astype(np.float32)
        
        
for i in range(0,41):
    if(max(test1[:,i])!=0):
        test1[:,i] = ((test1[:,i]-(min(test1[:,i])))/(max(test1[:,i])-(min(test1[:,i])))).astype(np.float32)
        ncol = train1.shape[1]

#X_train, X_test, Y_train, Y_test = train_test_split(train_scaled, target,train_size = 0.9, random_state = seed(2017))



### Define the encoder dimension
encoding_dim = 20

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
#print (encoding_dim)
print (input_dim)
#encoded1 = Dense(42*8, activation = 'relu')(input_dim)
#encoded2 = Dense(encoding_dim, activation = 'relu')(encoded1)
#
#decoded1 = Dense(42*8, activation = 'relu')(encoded2)
#decoded2 = Dense(ncol, activation = 'sigmoid')(decoded1)


#encoded1 = Dense(encoding_dim, activation = 'relu')(input_dim)
#decoded1 = Dense(ncol, activation = 'sigmoid')(encoded1)


"""6 Hidden Layers"""
encoded1 = Dense(42*8, activation = 'relu')(input_dim)
encoded2 = Dense(42*4, activation = 'relu')(encoded1)
encoded3 = Dense(encoding_dim, activation = 'relu')(encoded2)

decoded1 = Dense(42*8, activation = 'relu')(encoded3)
decoded2 = Dense(42*4, activation = 'relu')(decoded1)
decoded3 = Dense(ncol, activation = 'sigmoid')(decoded2)

autoencoder = Model(inputs = input_dim, outputs = decoded3)




## Combine Encoder and Deocder layers
#autoencoder = Model(inputs = input_dim, outputs = decoded2)

# Compile the Model
autoencoder.compile(optimizer = 'sgd', loss = 'mean_squared_error',metrics=['accuracy'])
autoencoder.summary()

autoencoder.fit(x=train1, y=train1, nb_epoch =20, batch_size = 32, shuffle = False, validation_data = (test1, test1))

#encoder = Model(inputs = input_dim, outputs = encoded2)
"""6 Hidden Layers"""
encoder = Model(inputs = input_dim, outputs = encoded3)
"""-------------------"""
encoded_input = Input(shape = (encoding_dim, ))



encoded_train = pd.DataFrame(encoder.predict(train1))
encoded_train = encoded_train.add_prefix('feature_')
encoded_train['label']=train_label



encoded_test = pd.DataFrame(encoder.predict(test1))
encoded_test = encoded_test.add_prefix('feature_')
encoded_test['label']=test_label




encoded_train.to_csv('train_encoded2.csv', index=False)
encoded_test.to_csv('test_encoded2.csv', index=False)


