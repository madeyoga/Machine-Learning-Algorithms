from mla.algos.regression.linear_model import LinearRegression
from mla.algos.metrics.score import root_mean_squared_error
from mla.algos.metrics.score import r2_score
import pandas as pd

## Fetch Data
data = pd.read_csv('Datasets/advertising_revenue_example.csv')
print(data)
X = data.Advertising
y = data.Revenue

## Init Model
lr = LinearRegression()

## Train Model
lr.fit(X, y)

## Predict
print(lr.predict([60, 50, 20, 40]))
predicted_y = lr.predict(X)

## root_mean_squared_error(actual_y, predicted_y)
rmse = root_mean_squared_error(y, predicted_y)
print("rmse: {}".format(rmse))

## r2_score(actual_y, predicted_y)
r2_sc = r2_score(y, predicted_y)
print("r2sc: {}".format(r2_sc))

## Visualize
import matplotlib.pyplot as plt
plt.plot(X, predicted_y, label="Regression Line")
plt.scatter(X, y, label="Data")
plt.xlabel('Advertising')
plt.ylabel('Revenue')
plt.legend()
plt.show()
