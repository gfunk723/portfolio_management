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
from calcCfg import *

print(portfolio)
print(symList)

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

def insertPortfolioQts(qtDict):
    pf = pd.read_csv('/Users/garrettfunk/Market/Management/EquitiesCalc/Data/Meta/Portfolio/portfolio_{}.csv'.format(portfolio),index_col='symbol')
    print(pf)
    ptfloSymList = list(qtDict.keys())
    qtDict = {sym:qtDict[sym] for sym in ptfloSymList}
    pf = pd.DataFrame(index=ptfloSymList, columns=pf.columns)
    pf.index = pf.index.rename('symbol')
    pf['qt'] = pd.Series(qtDict)
    priceSer = fetchIEXPortfolioPrices(ptfloSymList)
    pf['price'] = priceSer
    pf['value'] = pf['qt']*pf['price']
    pf['weight'] = pf['value']/liquidCash
    pf['date'] = datetime.date.today()
    print(pf)
    print('PORTFOLIO VALUE: {}'.format(round(pf['value'].sum(),2)))
    pf.to_csv('/Users/garrettfunk/Market/Management/EquitiesCalc/Data/Meta/Portfolio/portfolio_{}.csv'.format(portfolio))

def transformPctChange(df):
    for c in df.columns:
       df[c] = df[c].pct_change()
    return df

def transformLog(df):
    for c in df.columns:
       df[c] = df[c].apply(lambda x: np.log(1+x))
    return df

def historicalReturns(df,window):
    rf = df.resample(window).last()
    rf = transformPctChange(rf)
    rf = transformLog(rf)
    rf = rf.mean()
    print(rf)
    return rf

def historicaVols(df,windowSteps):
    vf = transformPctChange(df.copy())
    vf = transformLog(vf)
    vf = vf.std()*np.sqrt(windowSteps)
    print(vf)
    return vf

def makeCorrMatrix(df):
    cf = df.pct_change().apply(lambda x: np.log(1+x)).corr()
    cf.index.name = 'symbol'
    print(cf)
    cf.to_csv('Data/Meta/cov_matrix.csv')
    return cf

def makeCovMatrix(df):
    cf = df.pct_change().apply(lambda x: np.log(1+x)).cov()
    cf.index.name = 'symbol'
    print(cf)
    cf.to_csv('Data/Meta/cov_matrix.csv')
    return cf
    
def generatePortfolios(df,timeFrame,nPortfolios,rfRet):
    windowSteps = windowDict[timeFrame]
    retList = []
    volList = []
    weightsList = []
    nAssets = len(df.columns)
    print('HISTORICAL RETURNS')
    ret = historicalReturns(df,timeFrame)
    print('HISTORICAL VOL')
    histVols = historicaVols(df,windowSteps)
    print('CORR MATRIX')
    corrMatrix = makeCorrMatrix(df)
    print('COV MATRIX')
    covMatrix = makeCovMatrix(df)
    print(df)
    for i in range(nPortfolios):
        randMin = (nAssets*wMin)/(2-nAssets*wMin)
        weights = np.random.uniform(low=randMin, high=1.0, size=(nAssets,))
        weights = weights/np.sum(weights)
        weightsList.append(weights)
        returns = np.dot(weights, ret)
        retList.append(returns)
        portVar = covMatrix.mul(weights, axis=0).mul(weights, axis=1).sum().sum()
        vol = np.sqrt(portVar)*np.sqrt(windowSteps) # vol during window
        volList.append(vol)

    results = {'return':retList, 'volatility':volList}
    for i, sym in enumerate(df.columns.tolist()):
        results[sym] = [weights[i] for weights in weightsList]
    results  = pd.DataFrame(results)
    results['sharpe'] = (results['return'] - rfRet)/results['volatility']
    print('PORTFOLIO RESULTS')
    print(results.head())
    return results

def selectOptimalWeights(results,maxVol,plot,insertWeights):
    volSelection = results[results['volatility']<maxVol].sort_values(by=['return'],ascending=False).iloc[0,:].to_dict()
    volSelectWeights = {sym:round(volSelection[sym],4) for sym in symList}
    print('vol cutoff weights: {}'.format(volSelectWeights))

    maxSharpeSelection = results.sort_values(by=['sharpe'],ascending=False).iloc[0,:].to_dict()
    maxSharpeWeights = {sym:round(maxSharpeSelection[sym],4) for sym in symList}
    print('max sharpe weights: {}'.format(maxSharpeWeights))

    minVolSelection = results.sort_values(by=['volatility'],ascending=True).iloc[0,:].to_dict()
    minVolWeights = {sym:round(minVolSelection[sym],4) for sym in symList}
    print('min vol weights:    {}'.format(minVolWeights))
    
    if insertWeights:
        targetWeights = volSelectWeights
        print('target weights:     {}'.format(maxSharpeWeights))

        priceDict = fetchIEXPortfolioPrices(list(targetWeights.keys())).to_dict()
        print('prices:             {}'.format(priceDict))

        targetQts = {sym:math.floor(targetWeights[sym]*liquidCash/priceDict[sym]) for sym in targetWeights.keys()}
        print('target qts:         {}'.format(targetQts))
        insertPortfolioQts(targetQts)

    if plot:
        plt.scatter(results['volatility'], results['return'], marker='o', s=10, alpha=0.2)
        plt.scatter(volSelection['volatility'], volSelection['return'], color='b', marker='*', s=15)
        plt.scatter(maxSharpeSelection['volatility'], maxSharpeSelection['return'], color='r', marker='*', s=15)
        plt.scatter(minVolSelection['volatility'], minVolSelection['return'], color='g', marker='*', s=15)
        plt.show()

hf = makeValueHistory('close')
results = generatePortfolios(hf,timeFrame,100000,0.01)
selectOptimalWeights(results,0.20,True,True)