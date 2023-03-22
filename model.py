# DIABETES


#  consider  a dataset   pima  indian  diabetes  dataset
# Pregnancies: Number
# of
# times
# pregnant
# Glucose: Plasma
# glucose
# concentration
# a
# 2
# hours in an
# oral
# glucose
# tolerance
# test
# Blood
# Pressure: Diastolic
# blood
# pressure(mm
# Hg)
# Skin
# Thickness: Triceps
# skin
# fold
# thickness(mm)
# Insulin: 2 - Hour
# serum
# insulin(mu
# U / ml)
# BMI: Body
# mass
# index(weight in kg / (height in m) ^ 2)
# Diabetes
# Pedigree
# Function: Diabetes
# pedigree
# function
# Age: Age(years)
# Outcome: Class
# variable(0 or 1)

import pandas as pd
import numpy as np

columns = ["Pregnancies", "glucose", "bp", "skin", "insulin", "BMI", "Pedigree", "age", "result"]
df = pd.read_csv('./pima.csv', names=columns)

df

df.shape

df.head()

df.sample(5)

# display no of people with and without diabetes
df[df['result'] == 0]

df[df['result'] == 1]

N, Y = df['result'].value_counts()
print(N, Y)

import matplotlib.pyplot as plt

x = ["No diabtes", "diabetes"]
y = [N, Y]
plt.bar(x, y, color=["red", "green"])
plt.show()

# Display percentage wise in a pie chaart

a = plt.pie([N, Y], labels=["No diabates", "Diabetes"], autopct="%0.2f%%")

# Find Missing value is there or not
df.isnull().sum()
df.columns

print("Zero BP", len(df[df['bp'] == 0]))

df.columns

print("Zero glucosw", len(df[df['glucose'] == 0]))
print("Zero skin", len(df[df['skin'] == 0]))
print("Zero insulin", len(df[df['insulin'] == 0]))
print("Zero BMI", len(df[df['BMI'] == 0]))
print("Zero Pedigree", len(df[df['Pedigree'] == 0]))
print("Zero Age", len(df[df['age'] == 0]))

# fill the zzero values by median values
df['glucose'] = df['glucose'].replace(0, df['glucose'].median())

df['glucose']

df['bp'] = df['bp'].replace(0, df['bp'].median())
df['skin'] = df['skin'].replace(0, df['skin'].median())
df['insulin'] = df['insulin'].replace(0, df['insulin'].median())
df['BMI'] = df['BMI'].replace(0, df['BMI'].median())

print("Zero glucosw", len(df[df['glucose'] == 0]))
print("Zero skin", len(df[df['skin'] == 0]))
print("Zero insulin", len(df[df['insulin'] == 0]))
print("Zero BMI", len(df[df['BMI'] == 0]))
print("Zero Pedigree", len(df[df['Pedigree'] == 0]))
print("Zero Age", len(df[df['age'] == 0]))

df.sample(5)

# Divide the dataset into Input and Output
X = df.drop(columns='result')
Y = df.result

X

Y

# Split the data set into two parts :  training and testing
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=7)

X_train.shape

X_test.shape

Y_train.shape

Y_test.shape

# partcular range
# FEATURE SCALING : TO KEEP ONE UNIT
# Scale down  MinMaxScaler

from sklearn.preprocessing import MinMaxScaler

ms = MinMaxScaler()
X_train_ms = ms.fit_transform(X_train)
X_test_ms = ms.fit_transform(X_test)

X_train_ms

# Train the model using KNN algorithm

from sklearn.neighbors import KNeighborsClassifier

K = KNeighborsClassifier(n_neighbors=5)
K.fit(X_train_ms, Y_train)
import pickle
pickle.dump(K, open("model.pkl", "wb"))




