

# This module scrapes import data such as first price and etc to predict the
# close price

from difflib import SequenceMatcher
import pandas as pd
from urllib import *


def similar(str1, str2):
    return SequenceMatcher(None, str1, str2).ratio()


def find(name, dataset):
    length = len(dataset)
    for i in range(length):
        if name == dataset.iloc[i, 0]:
            return i
    return -1


def search(name, dataset):
    length = len(dataset)
    simList = []
    for i in range(length):
        if similar(name, dataset.iloc[i, 0]) > 0.6:
            simList.append(dataset.iloc[i, 0])
    return simList


def addSaham(name, url):
    urlDataset = open('urlDataset.csv', 'a')
    newRow = name + ',' + url
    urlDataset.write(newRow)
    urlDataset.close()


