""" LAST PREDICTOR """

# The machine learning module is for this program in RandomForest


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tsemodule5 as tm5

sns.set()


def dataCleaner(fileName):
    dataset = pd.read_csv(fileName)
    length = len(dataset)
    lastInfo = dataset.iloc[0, [5, 3, 4, 7]].values
    """
    if length >= 100:
        dataset = dataset.iloc[ :26, [2,3,4,5,7,10]]
    else:
        dataset = dataset.iloc[ :length-2, [2,3,4,5,7,10]]
    """

    dataset = dataset.iloc[:length - 2, [2, 3, 4, 5, 7, 10]]

    length = len(dataset)
    length -= 1
    x = dataset.iloc[1:, [1, 2, 4]]
    y = dataset.iloc[:length, [0, 3, 5]]
    dataset = np.append(y, x, axis=1)
    dataset = pd.DataFrame(dataset,
                           columns=['<FIRST>', '<CLOSE>', '<OPEN>(Yesterday close)', '<HIGH>', '<LOW>', '<VOL>'])
    return dataset, lastInfo


from sklearn.ensemble import RandomForestRegressor


def predictor(dataset, infoList):
    length = len(dataset)
    dataset = dataset.iloc[length::-1, 0:6]
    x = dataset.iloc[:, [0, 2, 3, 4, 5]].values
    y = dataset.iloc[:, 1:2].values
    y = y.reshape((len(y), 1))
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


def program():
    source = pd.read_csv('Source.csv')
    data = []
    length = len(source)
    for i in range(length):
        print(source.iloc[i, 0])
        data.append((best[i], run(source.iloc[i, 0] + '.csv', source.iloc[i, 1])))
    data = pd.DataFrame(data, columns=['Name', 'Final Price'])
    data.to_csv('result/result.csv')


# program()

best = ['اپال', 'اخابر', 'پارس', 'پارسان', 'جم', 'خودرو', 'رمپنا', 'شبندر', 'شتران', 'شپنا', 'شستا',
        'فارس', 'فخوز', 'فولاد', 'وبملت', 'وپارس', 'وپاسار',
        'وتجارت', 'وصندوق', 'ومعادن']

if __name__ == "__main__":
    # for stock in best:
    #     tm5.stock(stock)
    program()
