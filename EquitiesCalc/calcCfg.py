#!/usr/bin/env python
import subprocess
import os
import glob
import pandas as pd
import requests
token = os.environ['IEX_API_TOKEN']
windowDict = {'1m':21,
              '3m':63,
              '6m':126,
              '1y':252}

liquidDict = {'etrade':5000.,
              'ibkr':1401.}

portfolio = 'ibkr'
timeFrame = '3m'
wMin = 0.035
liquidCash = liquidDict[portfolio]


sf = pd.read_csv('/Users/garrettfunk/Market/Management/Pull/Data/Meta/Symbols/symbols_{}.csv'.format(portfolio),index_col='symbol')
sf = sf[sf['enable']==1]
symList = sf.index.tolist()

def fetchIEXPortfolioPrices(ptfloSymList):
    ptfloSymListStr = ','.join(ptfloSymList)
    print('Fetching prices: {}'.format(ptfloSymListStr))
    reqString = 'https://cloud.iexapis.com/stable/tops/last?symbols={}&token={}&format=json'.format(ptfloSymListStr,token)
    print(reqString)
    try:
        response = requests.get(reqString)
        priceSer = pd.DataFrame(response.json()).set_index('symbol')['price']
    except Exception as e: print(e)
    return priceSer