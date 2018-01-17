# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 10:01:07 2017

@author: RaghuramPC
"""

import pandas as pd
import numpy as np
from pandas_datareader import data
import statsmodels.api as sm
from statsmodels.api import add_constant
import datetime
import os
import time

class Strategy:
    def __init__(self, train_date_start, train_date_end, test_date_start, test_date_end):
        self.train_date_start = train_date_start; self.train_date_end = train_date_end; 
        self.test_date_start = test_date_start; self.test_date_end = test_date_end; 

    def compute_factors(self,n,m,L,total_df):
        total_df = total_df.sort_values(['Ticker','Date']).reset_index(drop=True)
        total_df['PM'] = total_df.groupby('Ticker')['Adj Close'].shift(1)/total_df.groupby('Ticker')['Adj Close'].shift(n)
        total_df['PRev'] = total_df.groupby('Ticker')['Adj Close'].shift(m)/total_df.groupby('Ticker')['Adj Close'].shift(1)
        total_df['log_RET'] = np.log(total_df['RET'])
        total_df['LVol'] = list(total_df.groupby('Ticker')['log_RET'].rolling(L).std())
        del total_df['log_RET']
        total_df = total_df.dropna(how='any').reset_index(drop=True)
        total_df = total_df[['Date','Ticker','Adj Close','Volume','PB','PCF','PE','PS','PM','PRev','LVol','RET']]
        return total_df
        
    def compute_factor_weights(self,total_df):
        train_df = total_df[(total_df['Date']>=pd.to_datetime(self.train_date_start)) & \
                    (total_df['Date']<=pd.to_datetime(self.train_date_end))].reset_index(drop=True)
        Y = train_df.iloc[:,-1].values.astype(float);
        X = train_df.iloc[:,4:-1].values.astype(float); X = add_constant(X, has_constant='add')
        result = sm.OLS(Y,X).fit()
        weights = result.params[1:]; weights = weights/sum(weights)
        return weights