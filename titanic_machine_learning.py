# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 17:23:29 2016

@author: praveen
"""

import pandas

titanic = pandas.read_csv("train.csv")

print(titanic.head(5))
print (titanic.describe())

titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())

print (titanic["Sex"].unique())

titanic.loc[titanic["Sex"] == "male","Sex"] = 0
titanic.loc[titanic["Sex"] == "female","Sex"] = 1

print (titanic["Sex"].unique())

titanic["Embarked"] = titanic["Embarked"].fillna("S")

print titanic["Embarked"].unique()

titanic.loc[titanic["Embarked"] == "S","Embarked"] = 0
titanic.loc[titanic["Embarked"] == "C","Embarked"] = 1
titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2

