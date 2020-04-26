# Required packages
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

dataset = pd.read_csv('crim.csv')
# x=dataset.drop( columns=['y', 'Date'])
# y=dataset['y']

cor=dataset.corr()
absoluteValuesCor = abs(cor["y"])
#Selecting highly correlated features
relatedVariables = absoluteValuesCor[absoluteValuesCor>0.5]
# print(relatedVariables)

#set the variables that have high correlation
x=dataset[['Total actions', 'Website actions', 'Directions actions']]
y=dataset['y']

###split our data set so that we can test our accuracy
X_train, X_test, y_train, y_test=train_test_split(x,y, test_size=0.30, random_state=16)
##define the model to be used
model=LinearRegression()
model.fit(X_train, y_train)

predictions=model.predict(X_test)
# print(predictions)

#Rsquared
print(model.score(X_test,y_test))
print(metrics.mean_squared_error(y_test,predictions))

# Saving model to disk
pickle.dump(model, open('model.pkl','wb'))

# Loading model to compare the results
modelpickel = pickle.load(open('model.pkl','rb'))
print(modelpickel.predict([[115, 50, 62]]))
