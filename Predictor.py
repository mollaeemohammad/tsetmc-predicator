

""" LAST PREDICTOR """

# The machine learning module is for this program in RandomForest



import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set()

def dataCleaner(fileName):
    dataset = pd.read_csv(fileName)
    dataset = dataset.iloc[ :100, [2,3,4,5,7,10]]
    length = len(dataset)
    length -= 1
    x = dataset.iloc[1:, [1,2,4]]
    y = dataset.iloc[:length, [0,3,5]]
    dataset = np.append(y, x, axis=1)
    dataset = pd.DataFrame(dataset, columns = ['<FIRST>', '<CLOSE>', '<OPEN>(Yesterday close)', '<HIGH>', '<LOW>', '<VOL>'])
    return dataset

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


def predictor(dataset, infoList):
    length = len(dataset)
    dataset = dataset.iloc[length::-1, 0:6]
    x = dataset.iloc[:, [0,2,3,4,5]].values
    y = dataset.iloc[:, 1:2].values
    y = y.reshape((len(y),1))
    regressor = RandomForestRegressor(n_estimators=200, random_state=0)
    regressor.fit(x, y)
    return regressor.predict([infoList])
    """
    length = len(dataset.iloc[:,-2])
    plt.scatter(np.array([i for i in range(length)]), dataset.iloc[:,-2], color='green')
    plt.plot(np.array([i for i in range(length)]), regressor.predict(x), color='red')
    plt.title('beautiful Predictoin ')
    plt.xlabel('Number Of Day')
    plt.ylabel('Last Price')
    """

#['<FIRST>', '<CLOSE>', '<OPEN>(Yesterday close)', '<HIGH>', '<LOW>', '<VOL>'])
#print(predictor(dataCleaner('Zanj.csv'),[8660, 8786, 9010, 8748, 5379000]))



