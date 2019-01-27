# Machine-Learning-Algorithms
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/66cbfc3c5cd54da1bb5a923a3afb35d1)](https://app.codacy.com/app/MadeYoga/ML-Algos?utm_source=github.com&utm_medium=referral&utm_content=MadeYoga/Machine-Learning-Algorithms&utm_campaign=Badge_Grade_Dashboard)
[![CodeFactor](https://www.codefactor.io/repository/github/madeyoga/Machine-Learning-Algorithms/badge)](https://www.codefactor.io/repository/github/madeyoga/Machine-Learning-Algorithms)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/MadeYoga/Machine-Learning-Algorithms/issues)
[![Discord Badge](https://discordapp.com/api/guilds/458296099049046018/embed.png)](https://discord.gg/Y8sB4ay)

Machine Learning Algorithms Implementation. Trying to create machine learning algorithms from scratch in simple way.
Hope this project could help those who wanted to learn machine learning agorithms.

## Requirements
- Python 3.x

## Dependencies
- Numpy
- matplotlib & pandas (to run examples)

## Installation
- Clone project
- Do `cd ...` to project dir
- Install module by `pip install .`
```git
C:\...\Machine-Learning-Algorithms>pip install .
```

## Usage Example
**Example's dataset can be found in examples folder `examples/Dataset`.**
Let's get started using pymla package, with a simple linear regression problem.
We are using simple `advertising_revenue_example.csv` Dataset,
And we want to predict the revenue from a given advertising data.

- First we need to load the dataset and use [pandas](https://github.com/pandas-dev/pandas) to fetch the data.
```py
import pandas as pd
dataset = pd.read_csv('Datasets/advertising_revenue_example.csv')
```

- Then, split the data to X and y
```py
X = dataset.advertising
y = dataset.revenue
```

- Build & train linear regression model.
```py
from pymla.model.linear_model import LinearRegression

linreg = LinearRegression()

# train model
linreg.fit(X, y)
```
- Predict the revenue.
```py
predicted_y = linreg.predict(X)
```

- Calculate error & score
```py
from pymla.metrics.score import root_mean_squared_error
from pymla.metrics.score import r2_score

## root_mean_squared_error(actual_y, predicted_y)
rmse = root_mean_squared_error(y, predicted_y)
print("rmse: {}".format(rmse))

## r2_score(actual_y, predicted_y)
r2_sc = r2_score(y, predicted_y)
print("r2sc: {}".format(r2_sc))
```

- Visualize result using [matplotlib](https://github.com/matplotlib/matplotlib)
```py
import matplotlib.pyplot as plt

plt.plot(X, predicted_y, label="Regression Line")
plt.scatter(X, y, label="Data")
plt.xlabel('Advertising')
plt.ylabel('Revenue')
plt.legend()
plt.show()
```

More examples can be found in [examples directory](https://github.com/MadeYoga/Machine-Learning-Algorithms/tree/master/examples).

## Available Algorithms
- Label Encoder
- MinMaxNormalization [0,1]
- Accuracy Score
  - Root Mean Squared Error 
  - R2 Score 
  - Accuracy Score 
- Linear Regression - Ordinary Least Square Method
- K-Nearest Neighbors
- Gaussian Naive Bayes

## To Do List
- Min Max Normalization  
  - [x1,x2]
- Metrics, Error & Accuracy
  - Confusion Matrix
  - F1-Score
- Model Selection, train test split & Cross Validation
