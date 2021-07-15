import subprocess
import os
import math
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
import scipy.stats as stats
import requests
import operator
import argparse
import pandas_market_calendars as mcal

from datetime import  datetime, timedelta
from calcCfg import *
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import LinearRegression
from scipy.fftpack import fft, rfft

def combineProcessedFiles():
    fileList = glob.glob('/Users/garrettfunk/Market/Management/Calc/Data/Stocks/Daily/Raw/*.csv')
    fileList = [file for file in fileList if ('high' in list(pd.read_csv(file,index_col=0,nrows=1).columns))]
    print('N Files: {}'.format(len(fileList)))
    dfList = [pd.read_csv(file,index_col='date',usecols=['date','high']).rename(columns={'high':file.split('/')[-1].split('_')[0]}) for file in fileList]
    df = pd.concat(dfList,ignore_index=False,sort=True,axis=1)
    df = df[symList]
    print(df.shape)
    for c in df.columns:
       df[c] = df[c].pct_change().apply(lambda x: np.log(1+x))
    # use prices to get 2 week volatility, 10 min candles
    vf = df[-10:].copy()
    vDict = {'value':'volatility'}
    for c in vf.columns:
        if vf[c].mean() != 0:
            vDict[c] = vf[c].std()/vf[c].mean()
        else:
            print(c)
    vf = pd.DataFrame([vDict])
    vf.to_csv('Data/Meta/2_week_vol.csv')
    # make covariance matrix
    print(df)
    df = df.corr()
    df.index.name = 'symbol'
    df.to_csv('Data/Meta/cov_matrix.csv')
    
combineProcessedFiles()
