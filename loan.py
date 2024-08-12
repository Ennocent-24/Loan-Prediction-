#!/usr/bin/env python3
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


data = pd.read_csv("loan.csv")

print(data.head())

from sklearn.model_selection import train_test_split
X = data.drop('Paid', axis=1)
Y = data['Paid']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, Y_train)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

predictions = model.predict(X_test)


sns.heatmap(confusion_matrix(Y_test, predictions), annot=True)
plt.show()

