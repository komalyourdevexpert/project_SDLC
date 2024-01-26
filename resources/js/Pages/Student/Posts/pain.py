import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

fname = r"C:\Users\komal\OneDrive\Desktop\BA\Real estate.csv"
data = pd.read_csv(fname)
x = data.iloc[:, 1:7]  # We're using the 2nd, 3rd, and 4th fields as predictors
y = data.iloc[:, 7]    # We're using the last field (sales) as our dependent variable

n_simul = 1000  # Number of simulation runs I want to perform
reg = LinearRegression()

R_squared_train = []
R_squared_test = []

for i in range(n_simul):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)  # Random splitting of the dataset
    reg.fit(x_train, y_train)  # Fitting the regression model

    R_squared_train.append(reg.score(x_train, y_train))  # Retrieving the R squared

    # Calculating out-of-sample R-squared using the formula R^2 = 1 - SSE/SST
    y_pred = reg.predict(x_test)  # Predicting the y_test using the test predictors

    SST_pred = sum((y_test - np.mean(y_test))**2)  # Total sum of squares for the test set
    SSE_pred = sum((y_test - y_pred)**2)          # Sum of squared errors for the test set
    R_squared_pred = 1 - SSE_pred / SST_pred       # Out-of-sample R-squared

    R_squared_test.append(R_squared_pred)

R_squared = pd.DataFrame({'Training': R_squared_train, 'Testing': R_squared_test})
stats = R_squared.describe()
print(stats)
