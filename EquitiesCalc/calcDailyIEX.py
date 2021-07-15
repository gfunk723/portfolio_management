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
from processCfg import *
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import LinearRegression
from scipy.fftpack import fft, rfft

parser = argparse.ArgumentParser()
parser.add_argument("--d", action="store_true")
parser.add_argument("-deploySym", action="append")
args = parser.parse_args()
if args.deploySym != None:
    symDeployList = args.deploySym[0]
    symDeployList = symDeployList.split(',')
    print(symDeployList)
    
# process params
candle_size = processParamCfgIntradayIEX['candle_size']
ma_windows = processParamCfgIntradayIEX['ma_windows']
fit_windows = processParamCfgIntradayIEX['fit_windows']
fft_windows = processParamCfgIntradayIEX['fft_windows']
fun_windows = processParamCfgIntradayIEX['fun_windows']
windowsMax = max(fit_windows)
volCol = processParamCfgIntradayIEX['volCol']
ls_periods = processParamCfgIntradayIEX['ls_periods']
split_frac = processParamCfgIntradayIEX['split_frac']
low_freq_cut = processParamCfgIntradayIEX['low_freq_cut']
adj_symbols = processParamCfgIntradayIEX['adj_symbols']
adj_cols = processParamCfgIntradayIEX['adj_cols']
t_days = processParamCfgIntradayIEX['t_days']
d_days = processParamCfgIntradayIEX['d_days']
d_periods = processParamCfgIntradayIEX['d_periods']

# tgt params
ma_window_tgt = processParamCfgIntradayIEX['ma_window_tgt']
fit_w_tgt = processParamCfgIntradayIEX['fit_w_tgt']
rise_factor_tgt = processParamCfgIntradayIEX['rise_factor_tgt']
lower_factor_tgt = processParamCfgIntradayIEX['lower_factor_tgt']
shift_factor_tgt = processParamCfgIntradayIEX['shift_factor_tgt']
smear_factor_tgt = processParamCfgIntradayIEX['smear_factor_tgt']

freqRangesDict = {}
for window in fft_windows:
    boundaries = [low_freq_cut]+[int(window*split_frac*i) for i in range(1,int(1./split_frac))] + [window+1]
    freqRangesDict[window] = [(boundaries[i],boundaries[i+1]) for i in range(len(boundaries)-1)]
print(freqRangesDict)

iexIntradayCols = ['date','minute','marketAverage','marketHigh','marketLow','marketOpen','marketClose','marketVolume','marketNumberOfTrades']
priceCols = ['marketAverage','marketHigh','marketLow','marketOpen','marketClose']
entryCols = ['marketVolume','marketNumberOfTrades']
marketCols = ['minute_count'] + priceCols + entryCols
    
def to_minute_count(minute):
    out = 60.*int(minute.split(':')[0]) + int(minute.split(':')[1])
    return out
    
def makeTS(df):
    df['ts'] = df['date'] + ' ' + df['minute']
    df['ts'] = pd.to_datetime(df['ts'])
    df['minute_count'] = df['minute'].apply(to_minute_count)
    df.drop(columns=['date','minute'],inplace=True)
    df = df.sort_values(by=['ts'],ascending=True)
    df = df.set_index('ts')
    return df
    
def getMarketDates(daysBack):
    date_today = datetime.today().date()
    start_date = date_today - timedelta(days = 2*daysBack)
    nyse = mcal.get_calendar('NYSE')
    dates = nyse.valid_days(start_date=start_date, end_date=date_today).strftime("%Y%m%d").tolist()[-daysBack:]
    print(dates)
    dates = [str(date) for date in dates]
    print(dates)
    return dates

def combineFiles(sym,train,daysBack = None):
    print(sym)
    try:
        fileList = glob.glob('/Users/garrett_funk/Market/Process/Data/Stocks/Intraday/Raw/{}_*.csv'.format(sym))
        print('N Files: {}'.format(len(fileList)))
        if not train:
            marketDatesBack = getMarketDates(daysBack)
            print(marketDatesBack)
            print('deploy dates')
            fileFilterList = [file for file in fileList if os.stat(file).st_size>0 and any([date in file for date in marketDatesBack])]
            print(fileFilterList)
        else:
            print('train all dates in training range')
            marketDatesBack = getMarketDates(t_days)
            fileFilterList = [file for file in fileList if os.stat(file).st_size>0 and any([date in file for date in marketDatesBack])]
        dfList = [pd.read_csv(file) for file in fileFilterList]
        dfList = [makeTS(df) for df in dfList if 'date' in list(df.columns)]
        print(len(dfList))
        dfList = [makeCandles(df,candle_size) for df in dfList if all([c in list(df.columns) for c in marketCols])]
        df = pd.concat(dfList,ignore_index=False,sort=True)[marketCols]
        df.sort_index(inplace=True,ascending=True)
        if not train:
            df = df[-1*int(d_periods):]
    except Exception as e: print(e)
    return df

def makeCandles(df,candleSize):
    df = df.groupby(pd.Grouper(freq='{}Min'.format(candleSize))).agg({
        "minute_count": "first",
        "marketAverage": "mean",
        "marketHigh":  "max",
        "marketLow":   "min",
        "marketOpen":  "first",
        "marketClose": "last",
        "marketVolume": "sum",
        "marketNumberOfTrades": "sum"})
    return df
    
def addTR(df):
    dayTrail = int(390/candle_size)
    oneWeekTrail = int(dayTrail*5)
    tf = df[['marketAverage','marketHigh','marketLow','marketOpen','marketClose']].copy()
    tf['TR1'] = abs( df['marketHigh'].rolling(dayTrail,center=False).max() - df['marketLow'].rolling(dayTrail,center=False).min() )
    tf['TR2'] = abs( df['marketHigh'].rolling(dayTrail,center=False).max() - df['marketClose'] )
    tf['TR3'] = abs( df['marketLow'].rolling(dayTrail,center=False).min() - df['marketClose'] )
    tf['TR'] = tf[['TR1','TR2','TR3']].max(axis=1)
    df['ATR'] = tf['TR'].ewm(span=oneWeekTrail).mean()
    df['ATR_fr'] = df['ATR'] / df['marketAverage']
    return df
     
def addMACD(df):
    lt = ma_windows[-1]
    st = ma_windows[-2]
    tf = df[['marketAverage']].copy()
    tf['ema_lt'] = tf['marketAverage'].ewm(span=lt).mean()
    tf['ema_st'] = tf['marketAverage'].ewm(span=st).mean()
    tf['MACD'] = (tf['ema_st'] - tf['ema_lt'])/tf['ema_lt']
    df['MACD'] = tf['MACD']
    return df
    
def linFit(df,val,window,trailing):
    model = RollingOLS(endog=df[val],exog=sm.add_constant(df[['x']]),window=window)
    fitModel = model.fit(params_only=True)
    if trailing:
        df['linfit_{}_{}'.format(window,val)] = fitModel.params['x'].fillna(0.)
    else:
        df['linfit_{}_{}'.format(window,val)] = fitModel.params['x'].shift(-int(window/2.)).fillna(0.)
    return df

def FFT(s):
    fftArr = np.abs(np.fft.fft(s))
    mag = np.linalg.norm(fftArr)
    if mag != 0:
        fftArr = fftArr / mag
    arrOut = list(fftArr)[:low_freq_cut]
    freqRanges = freqRangesDict[len(fftArr)]
    for freqRange in freqRanges:
        maxIndexInRange = freqRange[0]+np.argmax(fftArr[freqRange[0]:freqRange[1]])
        arrOut += [maxIndexInRange,fftArr[maxIndexInRange]]
    return np.array(arrOut)

def funnelCheck(a):
    aSign = np.sign(a)
    sChange = ((np.roll(aSign, 1) - aSign) != 0).astype(int)
    sChange = np.nonzero(sChange)[0]
    outList = sChange.tolist()[-6:]
    outList = [-1] * (6-len(outList)) + outList
    return np.array(outList)
    
def meanUpDown(s):
    arr = np.array(s)
    meanUp = np.sum(arr[arr>0.])/len(arr)
    meanDown = np.sum(arr[arr<0.])/len(arr)
    return np.array([meanUp,meanDown])
    
def rolling_windows(a, window):
    if window > a.shape[0]:
        raise ValueError('Specified window length of {} exceeds length of frame, {}'.format(window, a.shape[0]))
    a = a.values
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    shape = (a.shape[0] - window + 1, window) + a.shape[1:]
    strides = (a.strides[0],) + a.strides
    windows = np.squeeze(np.lib.stride_tricks.as_strided(a, shape=shape,strides=strides))
    # In cases where window == len(a), we actually want to "unsqueeze" to 2d.
    #     I.e., we still want a "windowed" structure with 1 window.
    if windows.ndim == 1:
        windows = np.atleast_2d(windows)
    return windows

def addIndex(sym,df,cols,train):
    if train:
        pf = pd.read_csv('/Users/garrett_funk/Market/Process/Data/Stocks/Intraday/Processed/{}_intraday_processed.csv'.format(sym),usecols = cols+['ts'])
    else:
        pf = pd.read_csv('/Users/garrett_funk/Market/Process/Data/Stocks/Intraday/Processed/Deployment/{}_intraday_processed_deployment.csv'.format(sym),usecols = cols+['ts'])
    pf['ts'] = pd.to_datetime(pf['ts'])
    pf = pf.set_index('ts')
    pf = pf.rename(columns={c:'{}_{}'.format(sym,c) for c in pf.columns})
    df = pd.merge(df,pf,how='left',on='ts')
    return df

def processIntradayIEXCSV(sym,train):
    df = combineFiles(sym,train,daysBack=d_days)
    df['x'] = np.array([i for i in range(df.shape[0])])
    
    #fill in null prices
    df[priceCols] = df[priceCols].fillna(method='ffill')
    df[entryCols] = df[entryCols].fillna(0.)

    if train:
        tf = df[['x','marketAverage']].copy()
        maCol = 'sma_{}'.format(ma_window_tgt)
        fitCol = 'linfit_{}_{}'.format(fit_w_tgt,maCol)
        tf[maCol] = tf['marketAverage'].rolling(ma_window_tgt,center=True).mean()
        tf = linFit(tf,maCol,fit_w_tgt,0)
        # up reversal
        tf['up_reversal'] = ((tf[fitCol].shift(-1)>0.) & (tf[fitCol]<=0.)).astype('int')
        tf['up_reversal'] = (tf['up_reversal'].rolling(ma_window_tgt,center=True).sum() > 0.)\
         & ((tf[maCol].shift(-shift_factor_tgt*ma_window_tgt) > rise_factor_tgt*tf['marketAverage']) | (tf[maCol].shift(-shift_factor_tgt*ma_window_tgt) > rise_factor_tgt*tf[maCol]))\
         & (tf[fitCol].shift(-shift_factor_tgt*ma_window_tgt).notnull())
        tf['up_reversal'] = tf['up_reversal'].rolling(smear_factor_tgt,center=True).sum() > 0.
        df['f_up_reversal'] = tf['up_reversal']
        # down reversal
        tf['down_reversal'] = ((tf[fitCol].shift(-1)<0.) & (tf[fitCol]>=0.)).astype('int')
        tf['down_reversal'] = (tf['down_reversal'].rolling(ma_window_tgt,center=True).sum() > 0.)\
         & ((tf[maCol].shift(-shift_factor_tgt*ma_window_tgt) < lower_factor_tgt*tf['marketAverage']) | (tf[maCol].shift(-shift_factor_tgt*ma_window_tgt) < lower_factor_tgt*tf[maCol]))\
         & (tf[fitCol].shift(-shift_factor_tgt*ma_window_tgt).notnull())
        tf['down_reversal'] = tf['down_reversal'].rolling(smear_factor_tgt,center=True).sum() > 0.
        df['f_down_reversal'] = tf['down_reversal']

        # plots
        # starts and stops
        tf['up_reversal_start'] = ((tf['up_reversal'].shift(1)==0.) & (tf['up_reversal']==1.)).astype('int')
        tf['up_reversal_end'] = ((tf['up_reversal'].shift(1)==1.) & (tf['up_reversal']==0.)).astype('int')
        upRevStartList = tf['x'][tf['up_reversal_start']==1].tolist()
        upRevEndList = tf['x'][tf['up_reversal_end']==1].tolist()
        upRevList = tf['x'][tf['up_reversal']==1].tolist()
        # create plot
        plt.figure(figsize=(100,10))
        plt.axhline(y=tf['marketAverage'].mean(), color='b')
        for rev in upRevStartList:
            plt.axvline(x=rev, linewidth=1, color='green', alpha=0.7)
        for rev in upRevEndList:
            plt.axvline(x=rev, color='red', linewidth=1, alpha=0.7)
        plt.plot(tf['x'],tf['marketAverage'], color='black')
        plt.plot(tf['x'],tf[maCol], color='orange')
        plt.savefig('PLOTS/ProcessIntradayPlots/{}_window_{}_fit_{}.pdf'.format(sym,ma_window_tgt,fit_w_tgt))
        plt.close()
    
    #widths
    df['width_hl_fr'] = (df['marketHigh'] - df['marketLow'])/df['marketLow']
    df['width_oc_fr'] = (df['marketClose'] - df['marketOpen'])/df['marketOpen']
    df['width_oc_hl_ratio'] = df['width_oc_fr']/df['width_hl_fr'].fillna(0.)
    df['open_candle_fr'] = ((df['marketOpen'] - df['marketLow'])/(df['marketHigh'] - df['marketLow'])).fillna(0.)
    df['close_candle_fr'] = ((df['marketClose'] - df['marketLow'])/(df['marketHigh'] - df['marketLow'])).fillna(0.)
    
    # ATR
    df = addTR(df)
    
    # stochastics
    for p in ls_periods:
        pSlow = p[0]
        pFast = p[1]
        df['ls_{}'.format(pSlow)] = (df['marketClose'] - df['marketLow'].rolling(pSlow).min())/(df['marketHigh'].rolling(pSlow).max() - df['marketLow'].rolling(pSlow).min())
        df['lsma_{}_{}'.format(pSlow,pFast)] = df['ls_{}'.format(pSlow)].rolling(pFast).mean()
        df['lsma_{}_{}_fsratio'.format(pSlow,pFast)] = df['ls_{}'.format(pSlow)] / df['lsma_{}_{}'.format(pSlow,pFast)]
        
    # MACD
    df = addMACD(df)
    
    # rolling rsi, means, and vols
    for w in ma_windows:
        df['sma{}_{}'.format(w,volCol)] = df[volCol].rolling(w).mean()
        df['smv{}_{}'.format(w,volCol)] = df[volCol].rolling(w).std()
        df['smvp{}_{}'.format(w,volCol)] = df[volCol].rolling(w).std()/df[volCol].rolling(w).mean()
        # rsi
        df_mud = pd.DataFrame(np.apply_along_axis(meanUpDown, 1, rolling_windows(df[volCol].pct_change(fill_method='ffill').fillna(0.), window=w)))
        df_zer = pd.DataFrame(0, index=np.arange(w-1), columns=df_mud.columns)
        df_mud = pd.concat([df_zer,df_mud],axis=0)
        df_mud = df_mud.set_index(df.index)
        df_mud = df_mud.rename(columns={0:'mean_up_pct_{}_{}'.format(volCol,w),1:'mean_down_pct_{}_{}'.format(volCol,w)})
        df = pd.concat([df,df_mud],axis=1)
        df['rsi{}_{}'.format(w,volCol)] = (100. - (100./(1.+(df['mean_up_pct_{}_{}'.format(volCol,w)]/df['mean_down_pct_{}_{}'.format(volCol,w)])))).fillna(0.)

    # pct change
    valsDiffPct = ['marketAverage']
    for val in valsDiffPct:
        df['{}_pct'.format(val)] = df[val].pct_change(fill_method='ffill').fillna(0.)

    # linear fits
    fitVals = ['marketAverage','marketVolume','marketNumberOfTrades']\
    + ['sma{}_{}'.format(w,volCol) for w in ma_windows]\
    + ['smv{}_{}'.format(w,volCol) for w in ma_windows]\
    + ['smvp{}_{}'.format(w,volCol) for w in ma_windows]\
    + ['width_hl_fr','width_oc_fr','width_oc_hl_ratio']\
    + ['open_candle_fr','close_candle_fr']\
    + ['ATR_fr']\
    + ['MACD']\
    + ['marketAverage_pct']\
    + ['rsi{}_{}'.format(w,volCol) for w in ma_windows]\
    + ['ls_{}'.format(p[0]) for p in ls_periods]\
    + ['lsma_{}_{}_fsratio'.format(p[0],p[1]) for p in ls_periods]
    print('Linear Fits')
    for val in fitVals:
        for window in fit_windows:
            print(val)
            print(window)
            df = linFit(df,val,window,1)

    # fast fourier transforms - add lin fit market average
    fftVals = ['sma{}_{}'.format(ma_windows[0],volCol)]\
            + ['linfit_{}_{}'.format(fit_windows[1],'sma{}_{}'.format(ma_windows[1],volCol))]\
            + ['lsma_{}_{}_fsratio'.format(p[0],p[1]) for p in ls_periods[:1]]
    print('Fast Fourier Transforms')
    for val in fftVals:
        for window in fft_windows:
            print(val)
            print(window)
            df_fft = pd.DataFrame(np.apply_along_axis(FFT, 1, rolling_windows(df[val], window=window)))
            df_zer = pd.DataFrame(0, index=np.arange(window-1), columns=df_fft.columns)
            df_fft = pd.concat([df_zer,df_fft],axis=0)
            df_fft = df_fft.set_index(df.index)
            df_fft = df_fft.rename(columns={c:'fft_{}_{}_{}'.format(val,window,i) for i,c in enumerate(df_fft.columns)})
            df = pd.concat([df,df_fft],axis=1)

    # funnel fits
    funnelVals = ['linfit_{}_{}'.format(fit_windows[0],'sma{}_{}'.format(ma_windows[1],volCol))]
    print('Funnel fits')
    for val in funnelVals:
        for window in fun_windows:
            print(val)
            print(window)
            df_fun = pd.DataFrame(np.apply_along_axis(funnelCheck, 1, rolling_windows(df[val], window=window)))
            df_zer = pd.DataFrame(0, index=np.arange(window-1), columns=df_fun.columns)
            df_fun = pd.concat([df_zer,df_fun],axis=0)
            df_fun = df_fun.set_index(df.index)
            df_fun = df_fun.rename(columns={c:'fun_{}_{}_{}'.format(val,window,i) for i,c in enumerate(df_fun.columns)})
            df = pd.concat([df,df_fun],axis=1)
            
    # Add other adjacent symbols
    if sym not in adj_symbols:
        for s in adj_symbols:
            df = addIndex(s,df,adj_cols,train)

    # Make meta data:
    metaFileName = '/Users/garrett_funk/Market/Process/Data/Meta/IntradayTemp/{}.csv'.format(sym)
    if os.path.exists(metaFileName):
        metaFrame = pd.read_csv(metaFileName,index_col='symbol')
        metaData = metaFrame[metaFrame.index==sym].to_dict('records')[0]
    else:
        metaData = {}
    metaData['symbol'] = sym
    metaData['calc_date'] = df.index[-1]
    # for removing old columns
    #for k in ['trend_{}'.format(180),'trend_week','trend_3hour']:
    #    if k in list(metaData.keys()):
    #        del metaData[k]
    if train:
        metaData['mean_rev_time'] = df['minute_count'][df['f_up_reversal']==True].mean()
    metaFrame = pd.DataFrame([metaData]).set_index('symbol')
    metaFrame.to_csv(metaFileName)
    print(metaFrame)

    # rm plain prices, keep marketAverage for meta data
    valsRm = ['marketOpen','marketHigh','marketLow','marketClose','x']\
            + [c for c in df.columns if 'sma' in c[:3]]\
            + [c for c in df.columns if 'smv' in c[:3] and 'smvp' not in c[:4]]
    df.drop(columns=valsRm,inplace=True)

    if not train:
        fileOut = '{}_intraday_processed_deployment.csv'.format(sym)
        df.to_csv('/Users/garrett_funk/Market/Process/Data/Stocks/Intraday/Processed/Deployment/{}'.format(fileOut))
    else:
        fileOut = '{}_intraday_processed.csv'.format(sym)
        df.to_csv('/Users/garrett_funk/Market/Process/Data/Stocks/Intraday/Processed/{}'.format(fileOut))
    print('Processed {}'.format(sym))

if args.d:
    if args.deploySym:
        print('Deployment process subset:')
        print(symDeployList)
        for sym in iexIntradayAdjSymbols + symDeployList:
            try:
                processIntradayIEXCSV(sym,0)
            except Exception as e: print(e)
                
    else:
        for sym in iexIntradaySymbolsAll:
            try:
                processIntradayIEXCSV(sym,0)
            except Exception as e: print(e)
else:
    print('processing: {}'.format(iexIntradaySymbolsAll))
    for sym in iexIntradaySymbolsAll:
        try:
            processIntradayIEXCSV(sym,1)
        except Exception as e: print(e)
