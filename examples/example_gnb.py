from mla.algos.Classification.naive_bayes import GaussianNB
from mla.algos.Preprocessing.LabelEncoder import LabelEncoder
import pandas as pd

df = pd.read_csv('Datasets/gwr.csv')
print(df.head())
X_train = df.loc[:, df.columns[:-1]]
y_train = df.loc[:, df.columns[-1:]]

X_train['gender'] = LabelEncoder().fit_transform(X_train['gender'].values)

gnb = GaussianNB()
gnb.fit(X_train.values, y_train.values)
