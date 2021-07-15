import subprocess
import os
import math
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import scipy.stats as stats
import scipy
import requests
import operator
import argparse
import pandas_market_calendars as mcal
import datetime
from scipy.stats import norm
from optionsCfg import *
from optionsCalc import *
from matplotlib.backends.backend_pdf import PdfPages
from statsmodels.distributions.empirical_distribution import ECDF

print(portfolio)
print(symList)

n = norm.pdf
N = norm.cdf

def makeValueHistory(value,daysBack):
    fileList = glob.glob('/Users/garrettfunk/Market/Management/EquitiesCalc/Data/Stocks/Daily/Raw/*.csv')
    fileList = [file for file in fileList if (value in list(pd.read_csv(file,nrows=1).columns))]
    print('N Files: {}'.format(len(fileList)))
    dfList = [pd.read_csv(file,index_col='date',usecols=['date',value]).rename(columns={value:file.split('/')[-1].split('_')[0]}) for file in fileList]
    df = pd.concat(dfList,ignore_index=False,sort=True,axis=1)
    df.index = pd.to_datetime(df.index)
    df = df[symList]
    df = df[-daysBack:]
    print(df.shape)
    print(df)
    return df

def historicalDailyLogReturns(df):
    rf = transformPctChange(df.copy())
    rf = transformLog(rf).mean()
    return rf

def transformPctChange(df):
    for c in df.columns:
       df[c] = df[c].pct_change()
    return df

def transformLog(df):
    for c in df.columns:
       df[c] = df[c].apply(lambda x: np.log(1+x))
    return df

def makeCorrMatrix(df):
    cf = df.pct_change().apply(lambda x: np.log(1+x)).corr()
    cf.index.name = 'symbol'
    print('CORRELATION')
    print(cf)
    cf.to_csv('Data/Meta/cov_matrix.csv')
    return cf

def makeCovMatrix(df):
    cf = df.pct_change().apply(lambda x: np.log(1+x)).cov()
    cf.index.name = 'symbol'
    print('COVARIANCE')
    print(cf)
    cf.to_csv('Data/Meta/cov_matrix.csv')
    return cf

def generateRandomNormals(nAssets,nEvents):
    A = np.random.normal(0, 1, (nAssets,nEvents))
    print(A)

def makeHistoPages(df, fileName, bins=50,):
    df = df[:-1]
    with PdfPages(fileName) as pdf:
        cols = df.columns
        pages = math.ceil(df.shape[1]/12.)
        h = 0
        for p in range(pages):
            plt.rcParams.update({'font.size': 6})
            fplots, plots = plt.subplots(4,3)
            for i in range(4):
                for j in range(3):
                    if h < df.shape[1]:
                        print(df.columns[h])
                        tempSer = df.iloc[:,h]
                        plots[i,j].set_xlim(-0.5,0.5)
                        plots[i,j].hist(np.array(tempSer),bins=bins)
                        plots[i,j].set_xlabel(cols[h])
                        h+=1
            fplots.tight_layout()
            pdf.savefig()
            plt.close()

def makeECDFPages(df, fileName, bins=50,):
    with PdfPages(fileName) as pdf:
        cols = df.columns
        pages = math.ceil(df.shape[1]/12.)
        h = 0
        for p in range(pages):
            plt.rcParams.update({'font.size': 6})
            fecdfs, ecdfs = plt.subplots(4,3)
            for i in range(4):
                for j in range(3):
                    if h < df.shape[1]:
                        print(df.columns[h])
                        tempSer = df.iloc[:,h]
                        ecdfTempSer = ECDF(tempSer)
                        ecdfs[i,j].plot(ecdfTempSer.x,ecdfTempSer.y,marker='.', linestyle='none',color='red', alpha=0.5)
                        ecdfs[i,j].set_xlabel(cols[h])
                        h+=1
            fecdfs.tight_layout()
            pdf.savefig()
            plt.close()

def correlatedRandNormals(nDays,corrWindow):
    print('HISTORICAL')
    hf = makeValueHistory('close',corrWindow)
    corr = makeCorrMatrix(hf).to_numpy()
    cov = makeCovMatrix(hf).to_numpy()
    print("CHOLESKY DECOMP L")
    L = scipy.linalg.cholesky(corr,lower=True)
    print(L)
    nAssets = hf.shape[1]
    normalsList = []
    for i in range(nDays):
            randNormals = np.random.normal(0, 1, size=(nAssets,))
            randNormalsCorr = np.matmul(L,randNormals)
            normalsList.append(randNormalsCorr)
    normalsCorrFrame = pd.DataFrame(normalsList,columns=hf.columns)
    print("CORRELATED RANDOM NORMALS")
    print(normalsCorrFrame)
    print('RANDOM NORMALS CORRELATION')
    print(normalsCorrFrame.corr())
    print('RANDOM NORMALS COVARIANCE')
    print(normalsCorrFrame.cov())
    return normalsCorrFrame

def simulateAssets(N,days,expiration,muReturn=False):
    returnDictList = []
    optionFrame, optionFrameSymGroup = collectOptionsList(expiration)
    symListOpt = list(optionFrameSymGroup.index)
    savePathsDict = {sym:[] for sym in symListOpt}
    for i in range(N):
        normalsCorrFrame = correlatedRandNormals(days,180)
        returnDict = {sym:None for sym in symListOpt}
        for sym in symListOpt:
            S0 = optionFrameSymGroup.loc[sym,'price']
            if muReturn == True:
                mu = optionFrameSymGroup.loc[sym,'histDailyReturn']
            else:
                mu = r
            sigma = optionFrameSymGroup.loc[sym,'dMinIV']
            print('symbol: {} S0: {} mu: {} sigma: {}'.format(sym,S0,mu,sigma))
            pathList = np.concatenate(([S0], np.zeros(normalsCorrFrame.shape[0])), axis=None)
            dt = 1
            for t in range(1,normalsCorrFrame.shape[0]+1):
                rand = normalsCorrFrame.loc[t-1,sym]
                pathList[t] = pathList[t-1] * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * rand)
            returnDict[sym] = pathList[-1]/pathList[0] - 1.
            savePathsDict[sym].append(pathList)
        returnDictList.append(returnDict)
    simReturnsFrame = pd.DataFrame(pd.DataFrame(returnDictList))
    print(simReturnsFrame)
    print(optionFrame)
    with PdfPages('underlying_simulations.pdf') as pdf:
        pages = math.ceil(len(savePathsDict)/6.)
        h = 0
        for p in range(pages):
            plt.rcParams.update({'font.size': 6})
            fpathplots, pathplots = plt.subplots(3,2)
            for j in range(3):
                for k in range(2):
                    if h < len(savePathsDict):
                        print(symListOpt[h])
                        for path in savePathsDict[symListOpt[h]]:
                            pathplots[j,k].plot(np.arange(len(path)),path)
                        pathplots[j,k].set_xlabel('Time t')
                        pathplots[j,k].set_ylabel('Price S')
                        pathplots[j,k].title.set_text(symListOpt[h])
                        h+=1
            fpathplots.tight_layout()
            pdf.savefig()
            plt.close()
    return simReturnsFrame,optionFrame
    # plt.show()

def varPortfolios(nP,N,days,expiration,replace=False):
    pctlList = [3,5,10,25,50,75,90,95,97]
    simReturnsFrame,optionFrame = simulateAssets(N,days,expiration)
    portfolioDictList = []
    for i in range(nP):
        portfolioDict = {}
        simPerformanceList = []
        sampleOptionFrame = optionFrame.sample(frac=1.,replace=replace)
        # place possible limit on sym, C, P
        sampleOptionFrame = sampleOptionFrame[sampleOptionFrame['cost'].cumsum() < liquidCash]
        if sampleOptionFrame.shape[0]>0:
            contractList = list(sampleOptionFrame.index)
            portfolioDict['contracts'] = contractList
            for i in simReturnsFrame.index:
                simRetFrame = pd.DataFrame(simReturnsFrame.iloc[i,:].T).rename(columns={i:'simReturn'})
                simRetFrame.index.name = 'symbol'
                simSampleOptionFrame = pd.merge(sampleOptionFrame,simRetFrame,on='symbol',how='left')
                simSampleOptionFrame['simOptValue'] = simSampleOptionFrame.apply(lambda x: valueOpt(x.simReturn,x.price,x.strike,x.side),axis=1)
                portfolioSimReturn = simSampleOptionFrame['simOptValue'].sum() / simSampleOptionFrame['cost'].sum()
                simPerformanceList.append(portfolioSimReturn)
                # print(simSampleOptionFrame)
            for pctl in pctlList:
                portfolioDict['pctl_{}'.format(pctl)] = np.percentile(np.array(simPerformanceList),pctl)
            portfolioDict['nAssets'] = len(contractList)
            portfolioDict['nDistinctUnderlying'] = len(set(list(sampleOptionFrame.symbol)))
            portfolioDict['ratioScore_75_50'] = portfolioDict['pctl_75']/(1. - portfolioDict['pctl_50'])
            portfolioDictList.append(portfolioDict)
    performanceTable = pd.DataFrame(portfolioDictList)
    performanceTable = performanceTable[(performanceTable['nDistinctUnderlying']>3)]
    performanceTable.sort_values(inplace=True,by=['pctl_50'],ascending=False)
    performanceTable.to_csv('performance_table.csv')
    print(performanceTable)

varPortfolios(300,1000,40,'20210820')