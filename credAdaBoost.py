import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

print("loading training data...")
raw_train = pd.read_csv("trainCred.csv")

train = raw_train.copy()

Xs_train = train.iloc[:, :-1].values
ys_train = train.iloc[:, -1].values

print("Running AdaBoost with incremental estimators and 5-fold CV")
xs = np.arange(0, 1000, 10)
xs[0] = 1
scores = []
max_score = -1
best_x = 0

for x in xs:
    clf = AdaBoostClassifier(n_estimators=x)
    cv_acc = cross_val_score(clf, Xs_train, ys_train, cv=5, n_jobs=-1)
    scores.append(cv_acc.mean())
    print(f"{x}: {cv_acc.mean():>4f}")
    if cv_acc.mean() > max_score:
        max_score = cv_acc.mean()
        best_x = x

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(xs, scores, color='tab:blue', label="training accuracy")
ax.legend()
ax.set_xlabel("estimators")
ax.set_ylabel("Accuracy")

plt.savefig("./images/adboost.png")

print(f"training final version with {best_x} classifiers...")
clf = AdaBoostClassifier(n_estimators=best_x)
clf = clf.fit(Xs_train, ys_train)

print("loading testing data...")
raw_test = pd.read_csv("testCred.csv")

test = raw_test.copy()

Xs_test = test.iloc[:, :-1].values
ys_test = test.iloc[:, -1].values

print("predicting from testing data...")
pred_test = clf.predict(Xs_test)

accuracy = accuracy_score(ys_test, pred_test)
print("Accuracy", accuracy)
