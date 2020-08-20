

# This module scrapes import data such as first price and etc to predict the
# close price

from difflib import SequenceMatcher
import pandas as pd
from urllib.request import urlopen


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


def makeNumber(number):
    number = list(number)
    length = len(number)
    delList = []
    count = 0
    for i in range(length):
        if number[i] == '٬':
            delList.append(i)
    for i in delList:
        number.pop(i-count)
        count += 1
    return int(''.join(number))


#['<FIRST>', '<CLOSE>', '<OPEN>(Yesterday close)', '<HIGH>', '<LOW>', '<VOL>'])
def getInfo(url):
    page = urlopen(url)
    html_bytes = page.read()
    html = html_bytes.decode("utf-8")
    # find first price
    # <span class="open" dir="ltr">۸٬۵۸۸</span>
    first_index = html.find('<span class="open" dir="ltr">')
    print(first_index)
    length = len('<span class="open" dir="ltr">')
    print(html[first_index: length])
    first_index = first_index + length
    end_index = html[first_index-1:].find('</span>') + first_index
    #FRIST = makeNumber(html[first_index, end_index])
    print(first_index, end_index)
    #print(html[first_index: end_index])
    # find open price
    # <span class dir="ltr">۸٬۸۵۳</span>
    #open_index = html.find('')
    









