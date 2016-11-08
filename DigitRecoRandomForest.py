# Imports

# pandas
import pandas as pd
from pandas import Series, DataFrame

# numpy, matplotlib, seaborn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# classifier
from sklearn.ensemble import RandomForestClassifier

# get digit train and test files
print "loading files..."
digit_train = pd.read_csv("train.csv")
digit_test = pd.read_csv("test.csv")
print "loading files finished"

digit_train_label = digit_train["label"]
digit_train = digit_train.drop("label", axis=1)

random_forest = RandomForestClassifier(n_estimators=100)
print "digit training started..."
random_forest.fit(digit_train, digit_train_label)
print "digit training finished"

print "digit prediction started..."
digit_test_pred = random_forest.predict(digit_test)
print "digit prediction finished"

print random_forest.score(digit_train, digit_train_label)

# make submission csv
image_id = []
for i, item in enumerate(digit_test_pred):
    image_id.append(i+1)

submit_results_rf = pd.DataFrame({
        "ImageId": image_id,
        "Label": digit_test_pred
    })

submit_results_rf.to_csv('submission_RandomForest.csv', index=False)
