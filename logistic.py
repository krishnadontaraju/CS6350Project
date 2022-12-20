import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

# import helper

import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names, but AdaBoostClassifier was fitted with feature names")

print("loading training data...")
raw_train = pd.read_csv("trainCred.csv")

# got info on categorical data handling from
# https://analyticsindiamag.com/complete-guide-to-handling-categorical-data-using-scikit-learn/

# determine categorical variables
# s = (raw_train.dtypes == 'object')
# object_cols = list(s[s].index)
#
# le = LabelEncoder()
train = raw_train.copy()

# LabelEncode the categorical variables
# for col in object_cols:
#     train[col+'num'] = le.fit_transform(train[col])
#     train.drop(col, axis=1, inplace=True)

# create Features/Labels dfs
Xs_train = train.iloc[:, :-1].values
ys_train = train.iloc[:, -1].values

# create AdaBoost Classifier and do 5-fold CV
print("20-fold cross-validation...")
lr = LogisticRegression(max_iter = 100000)
cv_acc = cross_val_score(lr, Xs_train, ys_train, cv=20, n_jobs=-1)
print(cv_acc.mean())

# train on full training dataset
print(f"training final version with full dataset...")
lr = LogisticRegression(max_iter = 100000)
lr = lr.fit(Xs_train, ys_train)

### Make Submission from test.csv
print("loading testing data...")
raw_test = pd.read_csv("testCred.csv")

test = raw_test.copy()
Xs_test = test.iloc[:, :-1].values
ys_test = test.iloc[:, -1].values

# LabelEncode the categorical variables
# for col in object_cols:
#     test[col+'num'] = le.fit_transform(test[col])
#     test.drop(col, axis=1, inplace=True)

#predict resulting values
print("predicting from testing data...")
pred_test = lr.predict(Xs_test)
accuracy = accuracy_score(ys_test, pred_test)
print("Accuracy",accuracy)
# data = {'ID': list(range(1, len(pred_test)+1)),
#         'Prediction': pred_test}
# out = pd.DataFrame(data)
#
# out.to_csv("./submissions/lr_submit.csv", index=False)