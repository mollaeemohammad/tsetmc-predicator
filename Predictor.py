

""" LAST PREDICTOR """

# The machine learning module is for this program in RandomForest



import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set()

def dataCleaner(fileName):
    dataset = pd.read_csv(fileName)
    length = len(dataset)
    lastInfo = dataset.iloc[0, [5,3,4,7]].values
    """
    if length >= 200:
        dataset = dataset.iloc[ :200, [2,3,4,5,7,10]]
    else:
        dataset = dataset.iloc[ :length-2, [2,3,4,5,7,10]]
    """
    dataset = dataset.iloc[ :length-2, [2,3,4,5,7,10]]
    length = len(dataset)
    length -= 1
    x = dataset.iloc[1:, [1,2,4]]
    y = dataset.iloc[:length, [0,3,5]]
    dataset = np.append(y, x, axis=1)
    dataset = pd.DataFrame(dataset, columns = ['<FIRST>', '<CLOSE>', '<OPEN>(Yesterday close)', '<HIGH>', '<LOW>', '<VOL>'])
    return dataset, lastInfo

def dataCleaner2(fileName):
    dataset = pd.read_csv(fileName)
    length = len(dataset)
    lastInfo = dataset.iloc[0, [5,3,4,7]].values
    if length >= 200:
        dataset = dataset.iloc[ :100, [2,3,4,5,7,10]]
    else:
        dataset = dataset.iloc[ :length-2, [2,3,4,5,7,10]]
    length = len(dataset)
    length -= 1
    x = dataset.iloc[1:, [1,2,4]]
    y = dataset.iloc[:length, [0,3,5]]
    dataset = np.append(y, x, axis=1)
    dataset = pd.DataFrame(dataset, columns = ['<FIRST>', '<CLOSE>', '<OPEN>(Yesterday close)', '<HIGH>', '<LOW>', '<VOL>'])
    return dataset, lastInfo

from sklearn.ensemble import RandomForestRegressor

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


def run(fileName, first_price):
    dataset, infoList = dataCleaner(fileName)
    infoList = np.append(np.array([first_price]), infoList)
    return str(predictor(dataset, infoList)[0])

def run2(fileName, first_price):
    dataset, infoList = dataCleaner2(fileName)
    infoList = np.append(np.array([first_price]), infoList)
    return str(predictor(dataset, infoList)[0])


def program():
    source = pd.read_csv('Source.csv')
    data = []
    data2 = []
    length = len(source)
    for i in range(length):
        data.append(run(source.iloc[i, 0]+'.csv', source.iloc[i, 1]))
        data2.append(run(source.iloc[i, 0]+'.csv', source.iloc[i, 1]))
    data = np.append(data, data2, axis = 0)
    data = pd.DataFrame(data, columns = ['Final Price All', 'FInal Price (100)'])
    data.to_csv('result\\result.csv')
program()
    
    
    
    
    