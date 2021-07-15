import subprocess
import os
import math
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import scipy.stats as stats
import requests
import operator
import argparse
import pandas_market_calendars as mcal
import datetime
from scipy.stats import norm
from optionsCfg import *

print(portfolio)
print(symList)

n = norm.pdf
N = norm.cdf

def makeValueHistory(value):
    fileList = glob.glob('/Users/garrettfunk/Market/Management/EquitiesCalc/Data/Stocks/Daily/Raw/*.csv')
    fileList = [file for file in fileList if (value in list(pd.read_csv(file,nrows=1).columns))]
    print('N Files: {}'.format(len(fileList)))
    dfList = [pd.read_csv(file,index_col='date',usecols=['date',value]).rename(columns={value:file.split('/')[-1].split('_')[0]}) for file in fileList]
    df = pd.concat(dfList,ignore_index=False,sort=True,axis=1)
    df.index = pd.to_datetime(df.index)
    df = df[symList]
    print(df.shape)
    print(df)
    return df

def historicalDailyReturns(df):
    rf = transformPctChange(df.copy())
    rf = transformLog(rf)
    rf = pd.DataFrame(rf.mean()).rename(columns={0:'histDailyReturn'})
    rf.index.rename('symbol',inplace=True)
    print(rf)
    return rf

def historicalReturns(df,window):
    rf = df.resample(window).last()
    rf = transformPctChange(rf)
    rf = transformLog(rf)
    rf = pd.DataFrame(rf.mean()).rename(columns={0:'histReturn'})
    rf.index.rename('symbol',inplace=True)
    print(rf)
    return rf

def transformPctChange(df):
    for c in df.columns:
       df[c] = df[c].pct_change()
    return df

def transformLog(df):
    for c in df.columns:
       df[c] = df[c].apply(lambda x: np.log(1+x))
    return df

def bs_price(cp_flag,S,K,T,r,v,q=0.0):
    d1 = (math.log(S/K)+(r+v*v/2.)*T)/(v*math.sqrt(T))
    d2 = d1-v*math.sqrt(T)
    if cp_flag == 'C':
        price = S*math.exp(-q*T)*N(d1)-K*math.exp(-r*T)*N(d2)
    else:
        price = K*math.exp(-r*T)*N(-d2)-S*math.exp(-q*T)*N(-d1)
    return price

def bs_vega(cp_flag,S,K,T,r,v,q=0.0):
    d1 = (math.log(S/K)+(r+v*v/2.)*T)/(v*math.sqrt(T))
    return S * math.sqrt(T)*n(d1)

def find_vol(target_value, call_put, S, K, T, r):
    MAX_ITERATIONS = 100
    PRECISION = 1.0e-4
    sigma = 0.5
    for i in range(0, MAX_ITERATIONS):
        price = bs_price(call_put, S, K, T, r, sigma)
        vega = bs_vega(call_put, S, K, T, r, sigma)

        price = price
        diff = target_value - price  # our root

        if (abs(diff) < PRECISION):
            return round(sigma,3)
        sigma = sigma + diff/vega # f(x) / f'(x)
    # value wasn't found, return best guess so far
    return round(sigma,3)

def valueOpt(ret,price,strikePrice,side):
    sideFactor = {'C':1,'P':-1}
    returnOpt = 100. * max(sideFactor[side]*((price * (1+ret))-strikePrice),0)
    return returnOpt

def collectOptionsList(expiration):
    sideFactor = {'C':1,'P':-1}
    optionFrame = getTWSOptions(symList,expiration)
    optionFrame['id'] = optionFrame.index.map(str)+'_'+optionFrame['side'].map(str)+'_'+optionFrame['strike'].map(str)
    optionFrame['symbol'] = optionFrame.index
    optionFrame = optionFrame.set_index('id')
    hf = makeValueHistory('close')
    dT = (datetime.datetime.strptime(expiration,'%Y%m%d').date() - datetime.datetime.today().date()).days
    dTy = dT/365.
    priceFrame = fetchIEXPrices(symList)
    print(priceFrame)
    optionFrame = pd.merge(optionFrame,priceFrame,how='left',on='symbol')
    optionFrame['cost'] = 100. * optionFrame['ask']
    ret = historicalDailyReturns(hf)
    optionFrame = pd.merge(optionFrame,ret,how='left',on='symbol')
    optionFrame['SKDiff'] = abs(optionFrame['price'] - optionFrame['strike'])
    optionFrame['IV'] = optionFrame.apply(lambda x: find_vol(x.ask, x.side, x.price, x.strike, dTy, r), axis=1)
    optionFrame = optionFrame.dropna()
    print(optionFrame)
    print(optionFrame.groupby('symbol')['IV'].idxmin())
    optionOtmIVGroup = optionFrame.loc[optionFrame.groupby('symbol')['IV'].idxmin()][['symbol','IV']].rename(columns={'IV':'minIV'}).set_index('symbol')
    print(optionOtmIVGroup)
    optionFrame = pd.merge(optionFrame,optionOtmIVGroup,how='left',on='symbol')
    optionFrame['dMinIV'] = optionFrame['minIV'] / math.sqrt(252)
    optionFrame = optionFrame.dropna().set_index('symbol')
    optionFrame.sort_values(inplace=True,by=['symbol','side','expiration','strike'])
    optionFrameSymGroup = optionFrame[['price','histDailyReturn','dMinIV']].groupby('symbol').first()
    optionFrame['id'] = optionFrame.index.map(str)+'_'+optionFrame['side'].map(str)+'_'+optionFrame['strike'].map(str)
    optionFrame['symbol'] = optionFrame.index
    optionFrame = optionFrame.set_index('id')
    optionFrame.to_csv('options_table.csv')
    print(optionFrame)
    return optionFrame,optionFrameSymGroup
