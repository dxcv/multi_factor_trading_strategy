# -*- coding: utf-8 -*-
"""
Created on Wed Dec 06 13:49:41 2017

@author: RaghuramPC
"""

import pandas as pd
import numpy as np
import math
import os
import time
import datetime
import Securities as se
import Strategy as st
import Portfolio as pf

nstocks = 100.0; init_portfolio_value = 10000000.0; avg_trading_vol = 1000000; tcost = 0.001
rf = 0.0237

def compute_port_vals(portfolio_df):
    portfolio_df = portfolio_df.dropna(how='any').reset_index(drop=True)
    pnl = portfolio_df['Portfolio Value'][portfolio_df.shape[0]-1] - init_portfolio_value
    adr = portfolio_df['RET'].mean(); sd = portfolio_df['RET'].std()
    sr = math.sqrt(252)*(adr-rf/252)/sd    
    mdd_end = np.argmax(np.maximum.accumulate(portfolio_df['Portfolio Value']) - portfolio_df['Portfolio Value'])
    mdd_start = np.argmax(portfolio_df['Portfolio Value'][:mdd_end])
    mdd = (portfolio_df['Portfolio Value'][mdd_end] - portfolio_df['Portfolio Value'][mdd_start])/portfolio_df['Portfolio Value'][mdd_start]
    return pnl, adr, sr, mdd
        
if __name__=='__main__':
    
    init_time = time.time()
    
    data_source = 'yahoo'
    start_date = '2010-1-1'
    end_date = '2015-7-31'
    mkt_cap_cutoff = 500000000
    nvalues = [i for i in range(5,6)]; mvalues = [i for i in range(20,41)];
    Lvalues = [i for i in range(20,21)]; Uvalues = [i for i in range(15,21)];
    weights = list()
    train_date_start = '2011-10-31'; train_date_end = '2014-10-31'; 
    test_date_start = '2014-11-1'; test_date_end = '2015-7-31'
    
    """ # These comments can be removed if you want to download the data
    nyse = se.Securities('ticker_universe.csv',mkt_cap_cutoff)
    total_df = nyse.get_stock_data(start_date,end_date,data_source)
    total_df.to_csv('all_stocks_mktcap.csv')
    
    market_data = nyse.get_market_data(start_date,end_date,data_source)
    market_data.to_csv('Market_Index.csv')
    market_data['Date'] = market_data.index; market_data = market_data.reset_index(drop=True)
    """
    
    # These two statements are used to load already downloaded stock and market index data
    market_data = pd.read_csv('Market_Index.csv',parse_dates=['Date']);
    total_df_main = pd.read_csv('all_stocks_mktcap.csv',parse_dates = ['Date']); del total_df_main['Unnamed: 0']
    #-----------------------------------------------------------------------
    
    strategy = st.Strategy(train_date_start,train_date_end,test_date_start,test_date_end)
    pnl_train = 0; adr_train = 0; sr_train = 0; mdd_train = 0
    count = 0
    for n in nvalues:
        for m in mvalues:
            for L in Lvalues:
                for U in Uvalues: 
                    total_df = strategy.compute_factors(n,m,L,total_df_main)
                    weights = strategy.compute_factor_weights(total_df)
                    total_df['MScore'] = np.matmul(total_df.iloc[:,4:-1],weights)
                    
                    portfolio = pf.Portfolio(nstocks,init_portfolio_value,avg_trading_vol,tcost)
                    
                    train_df = total_df[(total_df['Date']>=pd.to_datetime(train_date_start)) & \
                                        (total_df['Date']<=pd.to_datetime(train_date_end))].reset_index(drop=True)
                    portfolio_df_train = portfolio.compute_portfolio_normal(train_date_start,train_date_end,train_df,U)
                    portfolio_df_train['RET'] = portfolio_df_train['Portfolio Value']/portfolio_df_train['Portfolio Value'].shift(1) -1
                    pnl_temp, adr_temp, sr_temp, mdd_temp = compute_port_vals(portfolio_df_train)
                    if(sr_temp>sr_train):
                        pnl_train = pnl_temp; adr_train = adr_temp; sr_train = sr_temp; mdd_train = mdd_temp;
                        test_n = n; test_m = m; test_L = L; test_U = U;
                        portfolio_df_train.to_csv('portfolio_df_train.csv')
                    count += 1
                    
    total_df = strategy.compute_factors(test_n,test_m,test_L,total_df_main)
    weights = strategy.compute_factor_weights(total_df)
    total_df['MScore'] = np.matmul(total_df.iloc[:,4:-1],weights)
    
    #test_normal_strategy
    test_df = total_df[(total_df['Date']>=pd.to_datetime(test_date_start)) & \
                        (total_df['Date']<=pd.to_datetime(test_date_end))].reset_index(drop=True)
    portfolio_df_test = portfolio.compute_portfolio_normal(test_date_start,test_date_end,test_df,test_U)
    portfolio_df_test['RET'] = portfolio_df_test['Portfolio Value']/portfolio_df_test['Portfolio Value'].shift(1) -1
    portfolio_df_test.to_csv('portfolio_df_test.csv')
    pnl_test, adr_test, sr_test, mdd_test = compute_port_vals(portfolio_df_test)
    
    #---------------------------------------------------------------------------#
    U=test_U
    #train_beta_neutral_strategy
    train_df = total_df[total_df['Date']<=pd.to_datetime(train_date_end)].reset_index(drop=True)
    train_df = train_df.merge(market_data,on='Date',how='left').ffill()
    portfolio_df_train_beta_neutral = portfolio.compute_portfolio_beta_neutral(train_date_start,train_date_end,train_df,U)
    portfolio_df_train_beta_neutral['RET'] = portfolio_df_train_beta_neutral['Portfolio Value']/portfolio_df_train_beta_neutral['Portfolio Value'].shift(1) -1
    portfolio_df_train_beta_neutral.to_csv('portfolio_df_train_beta_neutral_testU.csv')
    pnl_train, adr_train, sr_train, mdd_train = compute_port_vals(portfolio_df_train_beta_neutral)
    
    #test_beta_neutral_strategy
    test_df = total_df[(total_df['Date']>=(pd.to_datetime(test_date_start)-datetime.timedelta(days=250))) & \
                    (total_df['Date']<=pd.to_datetime(test_date_end))].reset_index(drop=True)
    test_df = test_df.merge(market_data,on='Date',how='left').ffill()
    portfolio_df_test_beta_neutral = portfolio.compute_portfolio_beta_neutral(test_date_start,test_date_end,test_df,U)
    portfolio_df_test_beta_neutral['RET'] = portfolio_df_test_beta_neutral['Portfolio Value']/portfolio_df_test_beta_neutral['Portfolio Value'].shift(1) -1
    portfolio_df_test_beta_neutral.to_csv('portfolio_df_test_beta_neutral_testU.csv')
    pnl_test, adr_test, sr_test, mdd_test = compute_port_vals(portfolio_df_test_beta_neutral)
