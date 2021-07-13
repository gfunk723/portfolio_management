#!/usr/bin/env python
import subprocess
import os
from itertools import groupby
import pandas as pd
token = os.environ['IEX_API_TOKEN']

portfolio = 'ibkr'
# portfolio = 'etrade'

symList = pd.read_csv('/Users/garrettfunk/Market/Management/Pull/Data/Meta/Symbols/symbols_{}.csv'.format(portfolio),index_col='symbol').index.tolist()