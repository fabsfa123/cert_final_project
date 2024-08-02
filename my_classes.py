import pandas as pn        # data analysis and scientific computing
import numpy as np
from datetime import datetime
import math

import cufflinks as cf     # charting - productivity tool for ploty and pandas 
cf.set_config_file(offline=True)
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import plotly.graph_objects as go

from plotly.subplots import make_subplots
plt.style.use('seaborn')  
mpl.rcParams['font.family'] = 'serif'

import pylab 

from warnings import simplefilter
simplefilter(action="ignore", category=pn.errors.PerformanceWarning) # ignore performance warning


import random             # data science packages
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD

import sklearn
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.layers import Dropout
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.layers import SimpleRNN, LSTM, Dense, Dropout

import scipy.stats as stats
import statsmodels.api as sm
from boruta import BorutaPy # automated variable selection
from statsmodels.stats.outliers_influence import variance_inflation_factor

import talib

import itertools


class Instrument(object):
    '''
    Financial Instrument class
    
    Attributes:
            Inputs:
                file_name: str
                    full name of the txt file containing OHLCV time series to be backtested including extension
                    (.txt extension is required) 
                ticker: str
                    ticker's name
                description: str
                    description of intrument
            Data:
                data: DataFrame
                    raw OHLCV data
                resampled_data: DataFrame
                    resampled data
    Methods:
        get_data:
            imports data
        set_frequency:
            changes frequency of time series
        plot_daily_close:
            plot time series of close price
        create_descriptive_plots:
            show 4 plots for descriptive statistics
    '''
    def __init__(self,file_name,ticker,description):
        self.file_name = file_name
        self.ticker = ticker
        self.description = description
        self.get_data()
    
    def set_frequency(self,f):
        """
        Given a dataframe containing OHLCV intraday price data, it returns a Dataframe with a specified 
        frequency. 
        -------------------------------------------------------------------------------------
        -------------------------------------------------------------------------------------
    
        Inputs:
        -------
        f: string
            frequency
        """
        
        d = self.data
        
        out = pn.DataFrame()
        out['High']     = d.High.resample(f).max()
        out['Low']      = d.Low.resample(f).min()
        out['Open']     = d.Open.resample(f).first()
        out['Close']    = d.Close.resample(f).last()
        out['Volume']   = d.Volume.resample(f).sum()
        out.dropna(inplace=True) 
        
        self.resampled_data = out
        
        return
    
    def get_data(self):
        '''
        Import OHLCV time series
        '''
        _names= ['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume']
        self.data = pn.read_csv(self.file_name,delimiter=',',names= _names, 
                    index_col=0, parse_dates=True,infer_datetime_format=True)
        print (self.data.info())
    
    def plot_daily_close(self, second_instrument = None):
        """
        plot daily time series of close
        
        Inputs:
        -------
        
        second_instrument: Instrument Class
            use to add a new close line to the same plot
        
        """
        
        series1 = self.data.Close.resample('1d').last()
        _title =  'Historical Quotes' 
        _yaxis_title= self.ticker

        if second_instrument is not None:
            df = pn.DataFrame(index= series1.index,data=series1, columns=self.ticker)
            df[second_instrument.ticker] = second_instrument.data.Close.resample('1d').last()
            df.plot(secondary_ybool = second_instrument.ticker, figsize=(18, 8),title=_title)
        else:
            series1.plot(figsize=(18, 8),ylabel=_yaxis_title,title=_title)

    
    def create_descriptive_plots(self,d=None):
        """
        Creates 4 plots: histogram, box plot, qqplot, time series plot 
        -------------------------------------------------------------------------------------
        -------------------------------------------------------------------------------------

        Inputs:
        -------
        d: timeseries
        title: string    
        """ 
        
        if d is None:
            d = np.log(self.data['Close'] / self.data['Close'].shift(1))
        else:
            d = np.log(d['Close'] / d['Close'].shift(1))
        
        plt.figure(figsize=(18, 12))

        plt.subplot(221)
        plt.hist(d, bins=25, density=True, label=self.description)  
        plt.xlabel(self.description)
        plt.ylabel('Density')
        plt.title('Histogram')

        plt.subplot(222)
        dd = pn.DataFrame()
        dd[self.description] = d
        dd.boxplot(self.description) 
        plt.xlabel('data set')
        plt.ylabel('value')
        plt.title('Boxplot');

        plt.subplot(223)
        stats.probplot(d, dist="norm", plot=pylab)

        plt.subplot(224)
        plt.plot(d, lw=1.5, label=self.description)
        plt.legend(loc=0)
        plt.xlabel('time')
        plt.ylabel('value')
        plt.title('1st Data Set')
    