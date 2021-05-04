import pandas as pd
import numpy as np
import tensorflow as tf

#load data set
df=pd.read_csv('covtype.csv')

df

#class variable Cover type
df.Cover_Type

#store training set in train variable

train = df.iloc[: , :-1]

#training features are the column names of train data

training_features=train.columns

#split dataset into training and testing data using 80:20 ratio

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test= train_test_split(df[training_features],df[['Cover_Type']] , test_size=0.2, random_state=0)

#this step is used to one hot encode the output variable Cover type, because deep learning 
# models don't work on normal integer output variable data

from sklearn.preprocessing import OneHotEncoder

oc=OneHotEncoder(handle_unknown='ignore')

oc.fit(Y_test)
Y_test=oc.transform(Y_test).toarray()
oc.fit(Y_train)
Y_train=oc.transform(Y_train).toarray()

#this step is used to normalize the input data such that no two input features are very dissimilar in value range

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_test=sc.fit_transform(X_test)
X_train=sc.fit_transform(X_train)

#model definition, since keras is now incorporated in tensorflow 2.X.X we dont need to import keras anymore

model=tf.keras.Sequential()

# hidden layers
model.add(tf.keras.layers.Dense(units=100,activation='relu'))
tf.keras.layers.Dropout(0.2, noise_shape=None, seed=26)

model.add(tf.keras.layers.Dense(units=50,activation='relu'))
tf.keras.layers.Dropout(0.2, noise_shape=None, seed=26)

model.add(tf.keras.layers.Dense(units=100,activation='relu'))
tf.keras.layers.Dropout(0.2, noise_shape=None, seed=26)

#output layer

model.add(tf.keras.layers.Dense(units=7,activation='softmax'))

#compiling the model

model.compile(optimizer=tf.keras.optimizers.Adam(
    learning_rate=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
    name='Adam'), loss='categorical_crossentropy',metrics=(['accuracy']))

#training the model

model.fit(X_train,Y_train,batch_size=1000,epochs=40)

#predict on the test set

loss, acc= model.evaluate(X_test,Y_test)

print("Loss=",loss)
print("Test Accuracy",acc)

Y_pred=model.predict(X_test)

#this here is done because neural network models do not give you categorical values like in Y_test, 
#instead they give probabilities so we are going with the major probablities as the output class

Y_pred=(Y_pred>0.5)*1

from sklearn.metrics import classification_report,confusion_matrix
cr=classification_report(Y_test,Y_pred,digits=2,zero_division=False)
cr=cr.split(sep='\n')
for i in range(len(cr)):
    print(cr[i])
