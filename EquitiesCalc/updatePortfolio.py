import subprocess
import sys
import os
import math
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import json
import requests
import glob, os
import operator
import joblib
import argparse
import datetime
from calcCfg import *

print(timeFrame)
print(liquidCash)

def updatePrice():
    pf = pd.read_csv('/Users/garrettfunk/Market/Management/Calc/Data/Meta/Portfolio/portfolio_{}.csv'.format(portfolio),index_col='symbol')
    print(pf)
    print("_________________________")
    symQt = pf['qt'].to_dict()
    ptfloSymList = list(symQt.keys())
    priceSer = fetchIEXPortfolioPrices(ptfloSymList)
    pf['price'] = priceSer
    pf['value'] = pf['qt']*pf['price']
    pf['weight'] = pf['value']/liquidCash
    pf['date'] = datetime.date.today()
    print(pf)
    print('PORTFOLIO VALUE: {}'.format(round(pf['value'].sum(),2)))
    pf.to_csv('/Users/garrettfunk/Market/Management/Calc/Data/Meta/Portfolio/portfolio_{}.csv'.format(portfolio))

def showPortfolio():
    pf = pd.read_csv('/Users/garrettfunk/Market/Management/Calc/Data/Meta/Portfolio/portfolio_{}.csv'.format(portfolio),index_col='symbol')
    print(pf)

# showPortfolio()
updatePrice()

    