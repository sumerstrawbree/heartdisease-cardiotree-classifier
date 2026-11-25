import pandas as pd
import GWCutilities as util

pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth', None)

print("\n-----\n")

#Question: Given a patient's data, is this person likely to have heart disease?

#Create a variable to read the dataset
df = pd.read_csv("heartDisease_2020_sampling.csv")

print(
    "We will be performing data analysis on this Indicators of Heart Disease Dataset. Here is a sample of it: \n"
)

#Print the dataset's first five rows
print(df.head())

input("\n Press Enter to continue.\n")


#Data Cleaning
#Label encode the dataset
df = util.labelEncoder(df, ["HeartDisease", "Smoking", "AlcoholDrinking", "Sex", "PhysicalActivity", "GenHealth"])

print("\nHere is a preview of the dataset after label encoding. \n")
print(df.head())

input("\nPress Enter to continue.\n")

#One hot encode the dataset
df = util.oneHotEncoder(df, ["AgeCategory", "Race"]) 

print(
    "\nHere is a preview of the dataset after one hot encoding. This will be the dataset used for data analysis: \n"
)
print(df.head())

input("\nPress Enter to continue.\n")



#Creates and trains Decision Tree Model
from sklearn.model_selection import train_test_split
#holds the features (without the HeartDisease column)
X = df.drop("HeartDisease", axis = 1)
#holds the target class (contains the HeartDisease column)
y = df["HeartDisease"]

#X_test & y_test used to calculate a testing accuracy score
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 4)
print(X_train.head())

from sklearn.tree import DecisionTreeClassifier
#clf short for classifier
clf = DecisionTreeClassifier(max_depth = 8, class_weight = "balanced")
clf = clf.fit(X_train, y_train)

#Test the model with the testing data set and prints accuracy score
test_predictions = clf.predict(X_test)

print("TEST: ")
print(test_predictions)

from sklearn.metrics import accuracy_score
test_acc = accuracy_score(y_test,test_predictions)

print("The accuracy with the testing data set of the Decision Tree is: " + str(test_acc))
#run 1- 0.536, run 2- 0.554, run 3- 0.571

#Prints the confusion matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, test_predictions, labels =[1,0])
print("The confusion matrix of the tree is :")
print(cm)

#Test the model with the training data set and prints accuracy score
train_predictions = clf.predict(X_train)

from sklearn.metrics import accuracy_score
train_acc = accuracy_score(y_train,train_predictions)

print("The accuracy with the training data set of the Decision Tree is: " + str(train_acc))

input("\nPress Enter to continue.\n")


#Prints another application of Decision Trees and considerations
print("\nBelow is another application of decision trees and considerations for using them:\n")

print("A Decision Tree can be utilized to improve a process in the animation field by helping animation studios or artists recommend certain softwares or tools an artist can use based on their project idea.")

print("To ensure fairness, these factors should be considered in making sure the data can be applied to various artists: 1) bias 2) accuracy 3) diversity")

#Prints a text representation of the Decision Tree
print("\nBelow is a text representation of how the Decision Tree makes choices:\n")
input("\nPress Enter to continue.\n")

util.printTree(clf, X.columns)
