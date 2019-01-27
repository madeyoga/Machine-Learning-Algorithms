from pymla.model.naive_bayes import GaussianNB
from pymla.metrics.score import accuracy_score
from pymla.preprocessing.encoder import LabelEncoder
import pandas as pd

df = pd.read_csv('Datasets/gwr.csv')
print(df.head())
X_train = df.loc[:, df.columns[:-1]]
y_train = df.loc[:, df.columns[-1:]]

X_train['gender'] = LabelEncoder().fit_transform(X_train['gender'].values)

gnb = GaussianNB()
gnb.fit(X_train.values, y_train.values)

X_test = [[1, 175], [2, 130]]
y_test = ['high', 'high']

pred_y = gnb.predict(X_test)
print("predicted y: " + str(pred_y))
print("accuracy score: " + str(accuracy_score(pred_y, y_test)))
