# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 10:00:06 2017

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

class Securities:
    def __init__(self, file_tickers_universe,mkt_cap_cutoff):
        self.mkt_cap_cutoff = mkt_cap_cutoff
        self.tickers_universe = pd.read_csv(file_tickers_universe,parse_dates = ['ipo_date'])
        self.tickers_universe['mktshare'] = self.tickers_universe['mktshare'].str.replace(',', '').astype(float)
        self.tickers_universe = self.tickers_universe.rename(columns = {'ipo_date':'Date'})
        for i in range(self.tickers_universe.shape[0]):
            if(self.tickers_universe['ticker'][i][-2:]=='SS'):
                self.tickers_universe['ticker'][i] = self.tickers_universe['ticker'][i][:-2]+'ss'
    
    def get_stock_data(self, start_date, end_date, data_source):
        total_df = pd.DataFrame(columns=['Ticker','Date','PB','PCF','PE','PS','Adj Close','RET'])
        count=0
        for dirpath, dirnames, filenames in os.walk('./ss/'):
            for f in filenames:
                #if(count>1):
                #    continue
                if(f.endswith('.csv')):
                    stock_name = f[:-11]
                    req_df = pd.read_csv('./ss/'+f,parse_dates=['Date'])
                    req_df['Ticker'] = stock_name
                    try:
                        panel_data = data.DataReader(stock_name,data_source,start_date,end_date)[['Adj Close','Volume']]
                        panel_data = pd.DataFrame(panel_data).reset_index()
                    except:
                        continue
                    if(stock_name not in self.tickers_universe['ticker'].values):
                        continue
                    temp_df = self.tickers_universe[self.tickers_universe['ticker']==stock_name].reset_index()
                    if(temp_df['Date'][0] in panel_data['Date'].values):
                        mkt_cap = temp_df['mktshare']*panel_data[panel_data['Date']==temp_df['Date'][0]]['Adj Close'].values[0]
                    else: 
                        mkt_cap = temp_df['mktshare']*(panel_data['Adj Close'][0] + panel_data['Adj Close'][panel_data.shape[0]-1])/2.0
                        mkt_cap = mkt_cap.values[0]
                    if(mkt_cap<=self.mkt_cap_cutoff):
                        continue
                    req_df = pd.merge(req_df,panel_data,on='Date',how='outer').ffill()
                    req_df['RET'] = req_df['Adj Close']/req_df['Adj Close'].shift(1)
                    total_df = total_df.append(req_df)
                    print (count,f)
                    count += 1
        
        count=0
        print ("before sz")
        for dirpath, dirnames, filenames in os.walk('./sz/'):
            for f in filenames:
                #if(count>1):
                #    continue
                if(f.endswith('.csv')):
                    stock_name = f[:-11]
                    req_df = pd.read_csv('./sz/'+f,parse_dates=['Date'])
                    req_df['Ticker'] = stock_name
                    try:
                        panel_data = data.DataReader(stock_name,data_source,start_date,end_date)[['Adj Close','Volume']]
                        panel_data = pd.DataFrame(panel_data).reset_index()
                    except:
                        continue
                    if(stock_name not in self.tickers_universe['ticker'].values):
                        continue
                    temp_df = self.tickers_universe[self.tickers_universe['ticker']==stock_name].reset_index()
                    if(temp_df['Date'][0] in panel_data['Date'].values):
                        mkt_cap = temp_df['mktshare']*panel_data[panel_data['Date']==temp_df['Date'][0]]['Adj Close'].values[0]
                    else: 
                        mkt_cap = temp_df['mktshare']*(panel_data['Adj Close'][0] + panel_data['Adj Close'][panel_data.shape[0]-1])/2.0
                        mkt_cap = mkt_cap.values[0]
                    if(mkt_cap<=self.mkt_cap_cutoff):
                        continue
                    req_df = pd.merge(req_df,panel_data,on='Date',how='outer').ffill()
                    req_df['RET'] = req_df['Adj Close']/req_df['Adj Close'].shift(1)
                    total_df = total_df.append(req_df)
                    print (count,f)
                    count += 1
        
        return total_df
        
    def get_market_data(self,start_date,end_date,data_source):
        market_data = data.DataReader('000300.ss',data_source,start_date,end_date)['Adj Close']
        market_data = pd.DataFrame(market_data)
        market_data = market_data.rename(columns = {'Adj Close':'Mkt Close'})
        market_data['Mkt RET'] = market_data['Mkt Close']/market_data['Mkt Close'].shift(1)
        return 