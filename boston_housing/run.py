import numpy as np
import pandas as pd
import visuals as vs # Supplementary code
from sklearn.metrics import r2_score
from sklearn import cross_validation
from sklearn import svm
from sklearn.grid_search import GridSearchCV

# Pretty display for notebooks
# %matplotlib inline

# Load the Boston housing dataset
data = pd.read_csv('housing.csv')
prices = data['MEDV']
features = data.drop('MEDV', axis = 1)
    
# Success
print "Boston housing dataset has {} data points with {} variables each.".format(*data.shape)
minimum_price = prices.min()

# TODO: Maximum price of the data
maximum_price = prices.max()

# TODO: Mean price of the data
mean_price = prices.mean()

# TODO: Median price of the data
median_price = prices.median()

# TODO: Standard deviation of prices of the data
std_price = prices.std()

print "Result array:"
print "Min: {0}".format(minimum_price)
print "Max: {0}".format(maximum_price)
print "Mean: {0}".format(mean_price)
print "median: {0}".format(median_price)
print "Std: {0}".format(std_price)

minimum_price = np.min(prices)

# TODO: Maximum price of the data
maximum_price = np.max(prices)

# TODO: Mean price of the data
mean_price = np.mean(prices)

# TODO: Median price of the data
median_price = np.median(prices)

# TODO: Standard deviation of prices of the data
std_price = np.std(prices)

print "Result numpy:"
print "Min: {0}".format(minimum_price)
print "Max: {0}".format(maximum_price)
print "Mean: {0}".format(mean_price)
print "median: {0}".format(median_price)
print "Std2: {0}".format(std_price)



# print r2_score([3, -0.5, 2, 7, 4.2], [2.5, 0.0, 2.1, 7.8, 5.3]) 
# X_train, X_test, y_train, y_test = cross_validation.train_test_split(features, prices, test_size=0.2, random_state=0)
# print "r2_score: {0}".format(features.shape[0])

# parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
# svr = svm.SVC()
# clf = GridSearchCV(svr, parameters)
# clf.fit(features, prices)

# print clf.best_params_
