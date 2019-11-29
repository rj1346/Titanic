import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#importing training data
train= pd.read_csv('train.csv')
print(train.shape)
print(train.columns)

#Preparing X_train data set
X_train= train[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch','Embarked','Fare']]
print(X_train.head())
print(X_train.info())
print(X_train.isnull().sum())
sns.heatmap(X_train.isnull(), yticklabels=False)
plt.show()

X_train.Age= X_train['Age'].fillna(X_train.Age.mean())
X_train.Embarked= X_train['Embarked'].fillna(X_train.Embarked.mode()[0])

print(X_train.isnull().sum())
sns.heatmap(X_train.isnull(), yticklabels=False)
plt.show()

#Preparing y_train dataset
y_train= train[['Survived']]

#importing testing dataset
test= pd.read_csv('test.csv')
X_test= test[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch','Embarked','Fare']]

print(X_test.isnull().sum())
sns.heatmap(X_test.isnull(),yticklabels=False)
plt.show()

X_test.Age= X_test['Age'].fillna(X_test.Age.mean())
X_test.Fare= X_test['Fare'].fillna(X_test.Fare.mean())

print(X_test.isnull().sum())
sns.heatmap(X_test.isnull(),yticklabels=False)
plt.show()


#Creating Dummies
features= ['Sex','Embarked','Pclass']
for i in features:
    dummy= pd.get_dummies(X_train[i], drop_first=True)
    X_train= pd.concat([X_train,dummy], axis=1)
    X_train.drop(i, axis=1, inplace=True)

print(X_train.shape)

for i in features:
    dummy= pd.get_dummies(X_test[i], drop_first=True)
    X_test= pd.concat([X_test,dummy], axis=1)
    X_test.drop(i, axis=1, inplace=True)


#Applying Logistic Regression
from sklearn.linear_model import LogisticRegression
LogModel= LogisticRegression()
LogModel.fit(X_train,y_train)

y_pred= LogModel.predict(X_test)
y_pred=pd.DataFrame(y_pred)

#Submission
sample_submission= pd.read_csv('gender_submission.csv')
submission= pd.concat([sample_submission.PassengerId, y_pred], axis=1)
submission.columns=['PassengerId','Survived']
print(submission.head())

submission.to_csv('Prediction by LogReg.csv', index=None)