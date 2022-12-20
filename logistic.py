import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

print("loading training data...")
raw_train = pd.read_csv("trainCred.csv")

train = raw_train.copy()

Xs_train = train.iloc[:, :-1].values
ys_train = train.iloc[:, -1].values

print("5-fold CV")
lr = LogisticRegression(max_iter=100000)
cv_acc = cross_val_score(lr, Xs_train, ys_train, cv=5, n_jobs=-1)
print(cv_acc.mean())

print(f"training final version with full dataset")
lr = LogisticRegression(max_iter=100000)
lr = lr.fit(Xs_train, ys_train)

print("loading testing data")
raw_test = pd.read_csv("testCred.csv")

test = raw_test.copy()
Xs_test = test.iloc[:, :-1].values
ys_test = test.iloc[:, -1].values

print("predicting from testing data")
pred_test = lr.predict(Xs_test)
accuracy = accuracy_score(ys_test, pred_test)
print("Accuracy", accuracy)
