

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

X = pd.read_csv('train.csv')
X.head()
y = X.pop("Survived")
y.head()

numeric_variables = list(X.dtypes[X.dtypes != "object"].index)
X[numeric_variables].head

X["Age"].fillna(X.Age.mean(), inplace = True)
X.tail()
X[numeric_variables].head()

model = RandomForestClassifier(n_estimators = 500)
model.fit(X[numeric_variables],y)

print("Train_accuracy :: " , accuracy_score(y, model.predict(X[numeric_variables])))
test = pd.read_csv('test.csv')
test[numeric_variables].head()
test["Age"].fillna(test.Age.mean(), inplace = True)
test = test.fillna(test.mean())
test[numeric_variables].fillna(test.mean()).copy
y_pred = model.predict(test[numeric_variables])
y_pred

submission = pd.DataFrame({
        "PassengerId" : test["PassengerId"],
        "Survived" : y_pred
        })
submission.to_csv('Titanic.csv', index =False)