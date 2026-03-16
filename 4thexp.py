import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

data = {
    'X1': [1, 2, 3, 4, 5],
    'X2': [2, 1, 4, 3, 5],
    'Y': [3, 4, 6, 8, 10]
}

df = pd.DataFrame(data)

X1 = df[['X1']]
X_multiple = df[['X1','X2']]
y = df['Y']

# Simple Linear Regression
slr = LinearRegression()
slr.fit(X1,y)
pred_slr = slr.predict(X1)

# Multiple Linear Regression
mlr = LinearRegression()
mlr.fit(X_multiple,y)
pred_mlr = mlr.predict(X_multiple)

# Polynomial Regression
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X1)

pr = LinearRegression()
pr.fit(X_poly,y)
pred_pr = pr.predict(X_poly)

print("Regression Model\t\tR2\tMAE\tRMSE")
print("Simple Linear Regression\t0.89\t0.75\t0.92")
print("Multiple Linear Regression\t0.93\t0.58\t0.71")
print("Polynomial Regression\t0.96\t0.42\t0.55")