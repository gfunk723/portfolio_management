#!/user/local/bin/python
import subprocess
import os
import argparse
import json
import math
from decimal import Decimal
from datetime import  datetime, timedelta
import random
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
import requests
import time
import pandas_market_calendars as mcal
import pandas as pd
from pullCfg import *

print(symList)

def get_tickers():
    reqString = 'https://cloud.iexapis.com/stable/ref-data/iex/symbols?format=csv&token={}'
    df = pd.read_csv(reqString.format(token))
    df.to_csv('/Users/garrett_funk/Market/Pull/Data/Meta/tickers_iex.csv')
    tickers = df[~df['isEnabled'] == False].loc[:, 'symbol'].tolist()
    return tickers

def fetchIEXStatsMeta(sym,token):
    metaFileName = '/Users/garrettfunk/Market/Management/Pull/Data/Meta/DailyTemp/{}.csv'.format(sym)
    if os.path.exists(metaFileName):
        metaFrame = pd.read_csv(metaFileName,index_col='symbol')
        metaData = metaFrame[metaFrame.index==sym].to_dict('records')[0]
    else:
        metaData = {}
    print('Fetching financial stats: {}'.format(sym))
    reqString = 'https://cloud.iexapis.com/stable/stock/{}/advanced-stats?token={}'.format(sym,token)
    print(reqString)
    try:
        response = requests.get(reqString)
        outDict = json.loads(response.text)
        metaData.update(outDict)
        metaData['symbol'] = sym
        metaData['avg10ShareTurn'] = metaData['avg10Volume']/metaData['sharesOutstanding']
        metaData['avg30ShareTurn'] = metaData['avg30Volume']/metaData['sharesOutstanding']
        metaData['stats_date'] = datetime.today().date()
        metaFrame = pd.DataFrame([metaData]).set_index('symbol')
        metaFrame.to_csv(metaFileName)
        print(metaFrame)
    except Exception as e: print(e)
    
def runFetchAllSymbolsStatsMeta(selectList):
    for sym in [sym for sym in selectList]:
        fetchIEXStatsMeta(sym,token)

def fetchIEXCompanyMeta(sym,token):
    metaFileName = '/Users/garrettfunk/Market/Management/Pull/Data/Meta/DailyTemp/{}.csv'.format(sym)
    if os.path.exists(metaFileName):
        if 'company_info_date' in list(pd.read_csv(metaFileName,index_col=0,nrows=1).columns):
            print('company data filled for {}'.format(sym))
            return 0
        else:
            metaFrame = pd.read_csv(metaFileName,index_col='symbol')
            metaData = metaFrame[metaFrame.index==sym].to_dict('records')[0]
    else:
        metaData = {}
    print('Fetching company info: {}'.format(sym))
    reqString = 'https://cloud.iexapis.com/stable/stock/{}/company?token={}'.format(sym,token)
    print(reqString)
    try:
        response = requests.get(reqString)
        outDict = json.loads(response.text)
        metaData.update(outDict)
        metaData['symbol'] = sym
        metaData['company_info_date'] = datetime.today().date()
        metaFrame = pd.DataFrame([metaData]).set_index('symbol')
        metaFrame.to_csv(metaFileName)
        print(metaFrame)
    except Exception as e: print(e)

def runFetchAllSymbolsCompanyMeta(selectList):
    for sym in [sym for sym in selectList]:
        fetchIEXCompanyMeta(sym,token)

def fetchIEXDaily(sym,range,token):
    outfile = '/Users/garrettfunk/Market/Management/Pull/Data/Stocks/Daily/Raw/{}_{}_iex.csv'.format(sym,range)
    print('Fetching daily: {}'.format(sym))
    reqString = 'https://cloud.iexapis.com/stable/stock/{}/chart/{}?token={}&format=csv&chartIEXWhenNull=True'.format(sym,range,token)
    print(reqString)
    try:
        response = requests.get(reqString)
        print(response)
        file = open(outfile, "w")
        file.write(response.text)
        file.close()
        time.sleep(0.15)
    except Exception as e: print(e)

def runFetchAllSymbolsDaily(selectList,range):
    for sym in selectList:
        fetchIEXDaily(sym,range,token)


# runFetchAllSymbolsDaily(adjList+symDeployList,'3m')
# runFetchAllSymbolsDaily(symList,'3m')
runFetchAllSymbolsDaily(symList,'1y')
# runFetchAllSymbolsStatsMeta(symList)
# runFetchAllSymbolsCompanyMeta(symList)
