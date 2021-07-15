#!/usr/bin/env python
import subprocess
import os
import glob
import pandas as pd
import requests
import datetime
token = os.environ['IEX_API_TOKEN']
windowDict = {'1m':21,
              '3m':63,
              '6m':126,
              '1y':252}

liquidDict = {'etrade':5000.,
              'ibkr':1096.}

portfolio = 'ibkr'
timeFrame = '1m'
r = 0.0005
wMin = 0.035
liquidCash = liquidDict[portfolio]


sf = pd.read_csv('/Users/garrettfunk/Market/Management/Pull/Data/Meta/Symbols/symbols_{}.csv'.format(portfolio),index_col='symbol')
sf = sf[sf['enable']==1]
symList = sf.index.tolist()

def getTWSOptions(symList,expiration):
    print('Fetching options: {}, expiry: {}'.format(symList,expiration))
    optionFrame = pd.read_csv('/Users/garrettfunk/Market/Management/Pull/Data/Options/options_temp_{}.csv'.format(expiration))
    optionFrame.columns = [c.lower() for c in list(optionFrame.columns)[1:]] + [None]
    optionFrame.index.rename('symbol',inplace=True)
    optionFrame = optionFrame[optionFrame.index.isin(symList)]
    optionFrame = optionFrame.rename(columns={'p/c':'side'})
    optionFrame['expiration'] = datetime.datetime.strptime(expiration,'%Y%m%d').date()
    print(optionFrame)
    optionFrame.to_csv('options_all_contracts.csv')
    return optionFrame[['side','expiration','strike','ask','bid','last']].sort_values(by=['side','strike'],ascending=False)

def getIEXOptions(sym,expiration):
    print('Fetching options: {}, expiry: {}'.format(sym,expiration))
    reqString = 'https://cloud.iexapis.com/stable/stock/{}/options/{}?token={}&format=json'.format(sym,expiration,token)
    print(reqString)
    try:
        response = requests.get(reqString)
        optionFrame = pd.DataFrame(response.json()).set_index('symbol')
        optionFrame.to_csv('options_test.csv')
        optionFrame['expiration'] = datetime.datetime.strptime(expiration,'%Y%m%d').date()
        return optionFrame[['ask','bid','close','contractSize','strikePrice','expiration','side']].sort_values(by='strikePrice',ascending=False)
    except Exception as e: print(e)

def getIEXOptionDates(sym):
    print('Fetching options: {}'.format(sym))
    reqString = 'https://cloud.iexapis.com/stable/stock/{}/options?token={}&format=json'.format(sym,token)
    print(reqString)
    try:
        response = requests.get(reqString)
        datesList = pd.Series(response.json()).to_list()
        return datesList
    except Exception as e: print(e)

def fetchIEXPrices(ptfloSymList):
    ptfloSymListStr = ','.join(ptfloSymList)
    print('Fetching prices: {}'.format(ptfloSymListStr))
    reqString = 'https://cloud.iexapis.com/stable/tops/last?symbols={}&token={}&format=json'.format(ptfloSymListStr,token)
    print(reqString)
    try:
        response = requests.get(reqString)
        priceFrame = pd.DataFrame(response.json()).set_index('symbol')[['price']]
    except Exception as e: print(e)
    return priceFrame