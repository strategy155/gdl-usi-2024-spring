#!/usr/bin/env python
# coding: utf-8

# # Code

# ## Data

# In[177]:


# In[179]:


import pandas as pd
import numpy as np



def compute_diff_ts(stocknet_df: pd.DataFrame) -> pd.DataFrame :
    volume = stocknet_df.loc[:, (slice(None), "Volume")]
    stocknet_diff = stocknet_df.diff()[1:]
    stocknet_diff.loc[:, (slice(None), "Volume")] = volume[1:]
    return stocknet_diff

def minmax_bound_scaler(df,lower=.1,upper=.9, warn_threshold=10e-6):
    '''each colum is nomalized indepently of the others, values are mapped linearly in the range [lower,upper]'''
    df_norm = lower+(lower+1-upper)*(df-df.min())/(df.max()-df.min())

    assert df_norm.min().min()>=lower, "Something bad happened! Data below range"
    assert df_norm.max().max()<=upper, "Something bad happened! Data above range"
    
    invert_norm = lambda df_norm: ((df.max()-df.min())*(df_norm-lower)/(lower+1-upper))+df.min()
    err = (invert_norm(df_norm)-df).max().max()
    if err>warn_threshold: print(f"Warning, max. normalization inversion compound error {err}")
        
    return df_norm, invert_norm

def centered_bound_scaler(df,upper=.9, warn_threshold=10e-6):
    '''each column is normalized independenty of the others, values are mapped linearly in the range [-upper,upper], 0 is mapped to 0'''
    df_norm = upper*df/df.abs().max()
    assert df_norm.min().min()>=-upper-0.0000000000000001, "Something bad happened! Data below range"
    assert df_norm.max().max()<=upper+0.0000000000000001, "Something bad happened! Data above range"

    invert_norm = lambda df_norm: df.abs().max()*df_norm/upper
    err = (invert_norm(df_norm)-df).max().max()
    if err>warn_threshold: print(f"Warning, max. normalization inversion compound error {err}")
        
    return df_norm, invert_norm


# In[233]:


def unit_centered_scaler(df,upper=.9, warn_threshold=10e-6):
    '''each column is normalized based on it's measurement unit, values are mapped linearly in the range [-upper,upper], 0 is mapped to 0'''
    usd_cols = (slice(None), ["Open","High","Low","Close","Adj Close"])
    count_cols = (slice(None), "Volume")

    max_volume = df.loc[:, count_cols].abs().max().max()
    max_value = df.loc[:, usd_cols].abs().max().max()

    df_norm = df.copy()

    df_norm.loc[:, count_cols] = upper*df.loc[:, count_cols]/max_volume
    df_norm.loc[:, usd_cols] = upper*df.loc[:, usd_cols]/max_value

    assert df_norm.min().min()>=-upper-0.0000000000000001, "Something bad happened! Data below range"
    assert df_norm.max().max()<=upper+0.0000000000000001, "Something bad happened! Data above range"

    def invert_norm(df_norm):
        df = df_norm/upper
        df.loc[:, count_cols] = df.loc[:, count_cols]*max_volume
        df.loc[:, usd_cols] = df.loc[:, usd_cols]*max_value
        return df

    def invert_values(x):
        """in this case we also have one simple norm. funciton that inverts norm for all the stock values"""
        return max_value*x/upper
        
    err = (invert_norm(df_norm)-df).max().max()
    if err>warn_threshold: print(f"Warning, max. normalization inversion compound error {err}")
    
    return df_norm, invert_norm, invert_values


# In[231]:


def check_nans(stocknet_diff: pd.DataFrame):
    nans_per_stock = stocknet_diff.isnull().any(axis=0).groupby(level=0).sum()
    nanstocks = nans_per_stock.sort_values(ascending=False)[nans_per_stock.sort_values(ascending=False)>0]
    nanmatrix = stocknet_diff[nanstocks.keys()][stocknet_diff.isnull().any(axis='columns')]
    print(f"The nans are distributed over 6 stocks, affecting {nanmatrix.shape[0]} rows and {nanmatrix.shape[1]} columns, therefore it is advised to drop the {'columns' if nanmatrix.shape[0]>nanmatrix.shape[0] else 'columns'}")

