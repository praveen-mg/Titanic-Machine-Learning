# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 17:23:29 2016

@author: praveen
"""

import pandas
from sklearn.cross_validation import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score

titanic_train = pandas.read_csv("train.csv")
titanic_test = pandas.read_csv("test.csv")

def cleanData(data):
    #print(titanic.head(5))
    #print (titanic.describe())
    
    data["Age"] = data["Age"].fillna(data["Age"].median())
    
    #print (titanic["Sex"].unique())
    
    data.loc[data["Sex"] == "male","Sex"] = 0
    data.loc[data["Sex"] == "female","Sex"] = 1
    
    #print (titanic["Sex"].unique())
    
    data["Embarked"] = data["Embarked"].fillna("S")
    
    #print titanic["Embarked"].unique()
    #print "Unique values of class of travel:",titanic.Pclass.unique()
    data.loc[data["Embarked"] == "S","Embarked"] = 0
    data.loc[data["Embarked"] == "C","Embarked"] = 1
    data.loc[data["Embarked"] == "Q", "Embarked"] = 2
    return data
        



if __name__ == "__main__":
    
    data_train = cleanData(titanic_train)
    predictors = ['Sex','Pclass']
    feature = data_train[predictors]
    label = data_train['Survived']
    #print "Printing Survived"
    
    """
    Spliting the training data to test the prediction
    """
    feature_train, feature_test,label_train,label_test = train_test_split(feature,label, test_size = 0.4)
    
    clf = tree.DecisionTreeClassifier()
    
    clf.fit(feature_train,label_train)
    
    pred = clf.predict(feature_test)
    score = accuracy_score(pred,label_test)
    print "Accuracy value is:",score
    
    """
    Fit Decision tree to entire train data and predict the test data
     
    
    """
    data_test = cleanData(titanic_test)
    
    clf.fit(feature,label)
    pred = clf.predict(data_test[predictors])
    submission =  pandas.DataFrame({
                     "PassengerID": titanic_test["PassengerId"],
                      "Survived": pred})
                      
    submission.to_csv('submission.csv',index = False)






