import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

import time as t

csvCovid = pd.read_csv('Covid data.csv');
csvCovid.dropna()
CovidTotalRows = len(csvCovid)

for i in range(CovidTotalRows):
    if csvCovid.loc[i, 'DATE_DIED'] == '9999-99-99':
        csvCovid.loc[i, 'DATE_DIED'] = 0
    else:
        csvCovid.loc[i, 'DATE_DIED'] = 1

inFields = [
    'USMER',
    'MEDICAL_UNIT',
    'SEX',
    'PATIENT_TYPE',
    'INTUBED',
    'PNEUMONIA',
    'AGE',
    'PREGNANT',
    'DIABETES',
    'COPD',
    'ASTHMA',
    'INMSUPR',
    'HIPERTENSION',
    'OTHER_DISEASE',
    'CARDIOVASCULAR',
    'OBESITY',
    'RENAL_CHRONIC',
    'TOBACCO',
    'CLASIFFICATION_FINAL',
    'ICU',
    ]
outFields = ['DATE_DIED']
inX = csvCovid[inFields].astype('int')
outY = csvCovid[outFields].astype('int')

inXtrain, inXtest, outYtrain, outYtest = train_test_split(inX, outY)

svm = SVC();
startTime = t.time()

svm.fit(inXtrain.values, outYtrain.values.ravel())

print("Train ended in: ".format(t.time() - startTime))

startPredictionTime = t.time()
yPred = svm.predict(inXtest)

print("Prediction ended in: ".format(t.time() - startPredictionTime))

print("Accuracy of: ".accuracy_score(outYtest, yPred))
