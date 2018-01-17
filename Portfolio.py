# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 10:01:43 2017

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

class Portfolio:
    def __init__(self, nstocks, init_portfolio_value, avg_trading_vol, tcost):
        self.nstocks = nstocks; self.init_portfolio_value = init_portfolio_value;
        self.avg_trading_vol = avg_trading_vol; self.tcost = tcost;
    
    def compute_portfolio_normal(self,start_date,end_date,main_df,U):
        reb_days = int((pd.to_datetime(end_date) - pd.to_datetime(start_date)).days/U + 1)
        
        portfolio_df = pd.DataFrame(columns=['Date','Portfolio Value'])
        
        for i in range(reb_days):
            reb_date = pd.to_datetime(start_date) + datetime.timedelta(days=i*U)
            reb_date_end = reb_date + datetime.timedelta(days=U-1)
            if(reb_date_end>pd.to_datetime(end_date)):
                reb_date_end = pd.to_datetime(end_date)
            if(i==0):
                temp_df = main_df[(main_df['Date']>=reb_date) \
                            & (main_df['Date']<=reb_date_end)].reset_index(drop=True)
            else:
                temp_df = main_df[(main_df['Date']>=reb_date-datetime.timedelta(days=15)) \
                                & (main_df['Date']<=reb_date_end)].reset_index(drop=True)
            if(reb_date not in temp_df['Date'].values):
                reb_date = min((temp_df[temp_df['Date']>reb_date]['Date'].unique()))
                reb_date = pd.to_datetime(reb_date)
            vol_tickers = temp_df.groupby('Ticker').head(10).reset_index(drop=True).groupby('Ticker')['Volume'].mean()
            vol_tickers = list(vol_tickers[vol_tickers>self.avg_trading_vol].index)
            temp_df = temp_df[temp_df['Ticker'].isin(vol_tickers)].reset_index(drop=True)
            top_stocks = list(temp_df[temp_df['Date']==reb_date].sort_values('MScore',ascending=False)['Ticker'])[0:int(self.nstocks)]
            temp_df = temp_df[temp_df['Ticker'].isin(top_stocks)].reset_index(drop=True)
            temp_df = temp_df.sort_values(['Ticker','Date']).reset_index(drop=True)
            temp_df['Portfolio Value'] = temp_df['RET']
            if(i==0):
                init_value = self.init_portfolio_value/self.nstocks
                tcost_value = self.tcost*self.init_portfolio_value/self.nstocks
            else:
                init_value = portfolio_df['Portfolio Value'][portfolio_df.shape[0]-1]/self.nstocks
                common_stocks = list(set(top_stocks) & set(temp_stocks))
                drop_stocks = list(set(temp_stocks) - set(common_stocks))
                new_stocks = list(set(top_stocks) - set(common_stocks))
                tcost_value = self.tcost*(portfolio_df['Portfolio Value'][portfolio_df.shape[0]-1]* \
                                    (len(new_stocks) + len(drop_stocks))/self.nstocks)/self.nstocks
            temp_df.loc[temp_df.groupby('Ticker')['Portfolio Value'].head(1).index, 'Portfolio Value'] = init_value - tcost_value
            temp_df['Portfolio Value'] = temp_df.groupby('Ticker')['Portfolio Value'].cumprod()
            temp_df_portfolio = temp_df.groupby('Date')['Portfolio Value'].sum().reset_index()
            temp_df_portfolio = temp_df_portfolio[['Date','Portfolio Value']]
            portfolio_df = portfolio_df.append(temp_df_portfolio).reset_index(drop=True)
            #temp_stocks = top_stocks.copy()
            temp_stocks = top_stocks[:]
        
        return portfolio_df
    
    def compute_portfolio_beta_neutral(self,start_date,end_date,main_df,U):
        reb_days = int((pd.to_datetime(end_date) - pd.to_datetime(start_date)).days/U + 1)
        portfolio_df = pd.DataFrame(columns=['Date','Portfolio Value'])
        for i in range(reb_days):
            betas = list()
            reb_date = pd.to_datetime(start_date) + datetime.timedelta(days=i*U)
            reb_date_end = reb_date + datetime.timedelta(days=U-1)
            if(reb_date_end>pd.to_datetime(end_date)):
                reb_date_end = pd.to_datetime(end_date)
            if(i==0):
                temp_df = main_df[(main_df['Date']>=reb_date) \
                            & (main_df['Date']<=reb_date_end)].reset_index(drop=True)
            else:
                temp_df = main_df[(main_df['Date']>=reb_date-datetime.timedelta(days=15)) \
                                   & (main_df['Date']<=reb_date_end)].reset_index(drop=True)
            vol_tickers = temp_df.groupby('Ticker').head(10).reset_index(drop=True).groupby('Ticker')['Volume'].mean()
            vol_tickers = list(vol_tickers[vol_tickers>self.avg_trading_vol].index)
            temp_df = temp_df[temp_df['Ticker'].isin(vol_tickers)].reset_index(drop=True)
            if(reb_date not in temp_df['Date'].values):
                reb_date = min((temp_df[temp_df['Date']>reb_date]['Date'].unique()))
                reb_date = pd.to_datetime(reb_date)
            top_stocks = list(temp_df[temp_df['Date']==reb_date].sort_values('MScore',ascending=False)['Ticker'])[0:int(self.nstocks)]
            for j in range(len(top_stocks)):
                Y = main_df[(main_df['Ticker']==top_stocks[j]) & \
                             (main_df['Date']>=(reb_date-datetime.timedelta(days=100))) & \
                (main_df['Date']<=(reb_date-datetime.timedelta(days=1)))]['RET'].values.astype(float)
                if(Y.shape[0]==0):
                    betas.append(1)
                    continue
                X = main_df[(main_df['Ticker']==top_stocks[j]) & \
                             (main_df['Date']>=(reb_date-datetime.timedelta(days=100))) & \
                (main_df['Date']<=(reb_date-datetime.timedelta(days=1)))]['Mkt RET'].values.astype(float)
                X = add_constant(X, has_constant='add')
                result = sm.OLS(Y,X).fit()
                betas.append(result.params[1])
            avg_beta = np.mean(betas)
            temp_df = main_df[(main_df['Date']>=reb_date) & (main_df['Date']<=reb_date_end)].reset_index(drop=True)
            temp_df = temp_df[temp_df['Ticker'].isin(top_stocks)].reset_index(drop=True)
            temp_df = temp_df.sort_values(['Ticker','Date']).reset_index(drop=True)
            temp_df['Portfolio Value'] = temp_df['RET']
            if(i==0):
                init_value = self.init_portfolio_value/self.nstocks
                tcost_value = self.tcost*self.init_portfolio_value/self.nstocks
            else:
                init_value = portfolio_df['Portfolio Value'][portfolio_df.shape[0]-1]/self.nstocks
                common_stocks = list(set(top_stocks) & set(temp_stocks))
                drop_stocks = list(set(temp_stocks) - set(common_stocks))
                new_stocks = list(set(top_stocks) - set(common_stocks))
                tcost_value = self.tcost*(portfolio_df['Portfolio Value'][portfolio_df.shape[0]-1]* \
                                    (len(new_stocks) + len(drop_stocks))/self.nstocks)/self.nstocks
            temp_df.loc[temp_df.groupby('Ticker')['Portfolio Value'].head(1).index, 'Portfolio Value'] = init_value - tcost_value
            temp_df['Portfolio Value'] = temp_df.groupby('Ticker')['Portfolio Value'].cumprod()
            temp_df_portfolio = temp_df.groupby('Date')['Portfolio Value'].sum().reset_index()
            temp_df_portfolio = temp_df_portfolio[['Date','Portfolio Value']]
            temp_market = pd.DataFrame(temp_df['Mkt RET']) 
            temp_market = pd.DataFrame(temp_market['Mkt RET'].unique())
            temp_market.columns = ['Mkt RET']
            temp_market['Dollar Value'] = temp_market['Mkt RET']
            temp_market['Dollar Value'][0] = avg_beta*temp_df[temp_df['Date']==reb_date]['Mkt Close'][0]
            temp_market['Dollar Value'] = temp_market['Dollar Value'].cumprod()
            temp_df_portfolio['Portfolio Value'] = temp_df_portfolio['Portfolio Value'] - temp_market['Dollar Value']
            portfolio_df = portfolio_df.append(temp_df_portfolio).reset_index(drop=True)
            #temp_stocks = top_stocks.copy()
            temp_stocks = top_stocks[:]
            
        return portfolio_df