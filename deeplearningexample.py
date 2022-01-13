#importing packages
import pandas as pd
from keras.models import Sequential 
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt



#Load Concrete dataset

concrete = pd.read_csv('Concrete_Data.csv')
concrete.head()
concrete.info

X = concrete.copy()
X = X.drop('CompressiveStrength', axis = 1)
y = concrete.pop('CompressiveStrength')
input_shape = [X.shape[1]]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state= 45)
print("input shape:{}".format(input_shape))


#Building Network
model = Sequential()
model.add(Dense(512, activation = 'relu', input_shape = input_shape))
model.add(Dense(512, activation = 'relu'))
model.add(Dense(512, activation = 'relu'))
model.add(Dense(1))

#print (model.summary())

# Compile

model.compile( optimizer = 'adam', loss = 'mean_squared_error')
#opt = Adam(lr= 0.0015)
#model.compile( optimizer = opt, loss = 'mean_squred_error', metrics = ['mse'])
model.fit(X_train, y_train, epochs = 100)
