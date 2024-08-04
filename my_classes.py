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

#from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
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
                    full name of the txt file (including path) containing OHLCV time series to be back-tested including extension
                    (.txt extension is required) 
                ticker: str
                    ticker's name
                description: str
                    description of instrument
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
        Given a DataFrame containing OHLCV intraday price data, it returns a Dataframe with a specified 
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
            df = pn.DataFrame()
            df[self.ticker] = series1
            df[second_instrument.ticker] = second_instrument.data.Close.resample('1d').last()
            df.dropna(inplace = True)
            df.plot(secondary_y = second_instrument.ticker, figsize=(18, 8),title=_title)
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



class StrategyData(Instrument):
    '''
    Class containing methods and data containing all necessary data used for ML calibration, application and 
    strategy back-testing
    
    Attributes:
            inputs:
            -------------------------
            -------------------------
            data_name: str
                data description
            inst: Intrument object
                instrument to be back-tested
            freq: str
                frequency of data
            vix: Instrument object
                object containing vix data - if None, vix is not added as a predictor
            
            data:
            ------------------------
            ------------------------
            cleansed_data: DataFrame
                data aggregated at a given frequency, with vix added as a predictor, 
                and adjusted for trading hours and missing values
            features_data: DataFrame:
                cleansed data with features
            labels_data: DataFrame
                cleansed data with Labels
            labels_train, labels_test, features_train,features_test: DataFrames
                labels and features split in training and test. features are standardised
                
        Methods:
            fill_OHLCV_missing_data: 
                fills missing values in a OHLCV DataFrame
            add_vix:
                merge ViX OHLCV data with main instrument to be predicted
            set_frequency:
                resample DataFrame with one or two OHLCV instruments 
            number_of_bars_per_day:
                returns max number of OHLCV bars per day in a DataFrame
            create_features:
                adds multiple features to a DataFrame for intraday ML forecasting
            create_labels:
                create ML labels
            split_train_test:
                split features and labels into train and test
            standardise:
                standardise ML data
            split_data_standardise:
                split features and labels into training and test and then standardise features
            set_trading_hours:
                delete bars outside of a specified time span to align ML calibration with application of trading
                strategy
            
    '''
    def __init__(self,inst,freq,data_name='',vix=None):
        self.data_name = data_name
        self.inst = inst
        self.freq = freq
        self.vix = vix
        self.add_vix()
        self.set_frequency()
    
    def fill_OHLCV_missing_data(self,d,ticker):
        '''
        fills missing values in a OHLCV DataFrame
        
        Inputs:
            d: DataFrame
            ticker: str
        Output: DataFrame
            
        '''
        
        d['Volume_'+ticker]=d['Volume_'+ticker].fillna(0.)         # fill missing volume with 0 
        d['Close_'+ticker].fillna(method='ffill',inplace=True)     # forward fill Close
        d['Open_'+ticker].fillna(d['Close_'+ticker],inplace=True)  # fill missing Open with forward filled Close
        d['Low_'+ticker].fillna(d['Close_'+ticker],inplace=True)   # fill missing Low with forward filled Close
        d['High_'+ticker].fillna(d['Close_'+ticker],inplace=True)  # fill missing High with forward filled Close
        
        return d
    
    def add_vix(self):
        '''
        merge ViX OHLCV data with main instrument to be predicted
        
        '''
        # merge with vix if applicable
        if self.vix is not None:
            _data = pn.merge(self.inst.data,self.vix.data,left_index=True,
                                            right_index=True, how='left', 
                                            suffixes=('_'+self.inst.ticker,'_'+self.vix.ticker))
            _data = self.fill_OHLCV_missing_data(_data,self.vix.ticker)
            _data.dropna(inplace=True)
            self.cleansed_data = _data
        else:
            self.cleansed_data = self.inst.data.add_suffix('_'+self.inst.ticker)
    
    
    def number_of_bars_per_day(self,d):
        '''
        returns max number of records per day in a dataframe (d)
        '''
        return d.groupby(d.index.date).count().max().max()

    def set_frequency(self,start_time="00:00:00",end_time="23:59:59",exclude_weekends=True):
        """
        this method overrides instrument class method to handle having two OHLCV instruments merged in the 
        same DataFrame
        
        Given a DataFrame containing OHLCV intraday price data, it returns a DataFrame with a specified 
        frequency. It is possible to remove bars outside a specific time range and exclude weekends 
        -------------------------------------------------------------------------------------
         -------------------------------------------------------------------------------------
         
        Inputs:
        -------
        d: DataFrame
            OHLCV dataframe with 2 instruments
        start_time: string
            start of official trading session - %H:%M:%S" format - 24 hour clock- (default="00:00:00") 
        end_time: string
            end of official trading session - %H:%M:%S" format - 24 hour clock - (default="23:59:99")
        exclude_weekends: bool 
            if True, weekends are removed
        """
        start_time = datetime.strptime(start_time,"%H:%M:%S").time()
        end_time = datetime.strptime(end_time,"%H:%M:%S").time()
        highest_dayofweek = 5 if exclude_weekends else 8
        
        d = self.cleansed_data
        _ticker = self.inst.ticker 
        
        
        d = d[(d.index.time >= start_time) & (d.index.time < end_time) & (d.index.dayofweek < highest_dayofweek)]
        
        out = pn.DataFrame()
        
        #instrument
        out['High_'+_ticker]     = d['High_'+_ticker].resample(self.freq).max()
        out['Low_'+_ticker]      = d['Low_'+_ticker].resample(self.freq).min()
        out['Open_'+_ticker]     = d['Open_'+_ticker].resample(self.freq).first()
        out['Close_'+_ticker]    = d['Close_'+_ticker].resample(self.freq).last()
        out['Volume_'+_ticker]   = d['Volume_'+_ticker].resample(self.freq).sum()
        
        #vix
        if self.vix is not None:
            _vix =self.vix.ticker
            out['High_'+_vix]     = d['High_'+_vix].resample(self.freq).max()
            out['Low_'+_vix]      = d['Low_'+_vix].resample(self.freq).min()
            out['Open_'+_vix]     = d['Open_'+_vix].resample(self.freq).first()
            out['Close_'+_vix]    = d['Close_'+_vix].resample(self.freq).last()
            out['Volume_'+_vix]   = d['Volume_'+_vix].resample(self.freq).sum()
        
        out.dropna(inplace=True) 
        
        self.cleansed_data = out
    
    def RollingOBV(self, data, ticker, timeperiod):
        '''
        Calculate rolling On-Balance-Volume Indicator (OBV)
        
        Inputs:
            data: DataFrame
                DataFrame containing inputs for OBV (Close and Volume)
            ticker: str 
                ticker name
            timeperiod: int
                length of rolling window
        '''
        
        df = pn.DataFrame(index=data.index)
        df['up_bar'] = np.where(data['Close_'+ticker] > data['Close_'+ticker].shift(1),1.,-1.)
        df['volume'] = data['Volume_'+ticker]
        
        return df.product(axis=1).rolling(timeperiod).sum()
    
    
    
    
    def create_features(self,n=[2,5,10,25,50,100,250]):
        """
        Given a DataFrame with OHLCV price data, it returns a Dataframe including multiple
        features for intra day ML forecasting
        -------------------------------------------------------------------------------------
        -------------------------------------------------------------------------------------
        
        Inputs:
        -------
        d: DataFrame
            DataFrame with a DateTime index containing OHLVC data for one or multiple instruments 
            labelled as "Close_"&ticker
        n:  list of ints
            length/s of rolling window/s
        """
        
        d = self.cleansed_data
        tickers = [self.inst.ticker]
        
        out = pn.DataFrame(index=d.index)
        
        if self.vix is not None:
            tickers.insert(0,self.vix.ticker)
        
        bars_per_day = self.number_of_bars_per_day(d)
        
        for t in tickers:
            if t in 'VX':
                out[f"Close_{t}"] = d['Close_'+t]
            out[f"d_1_{t}"] = np.where(d['Close_'+t] - d['Close_'+t].shift(1) > 0, 1, 0)
            out[f"hl_1_{t}"] =  d['High_'+t] - d['Low_'+t]
            out[f"hc_1_{t}"] =  d['High_'+t] - d['Close_'+t]
            out[f"cl_1_{t}"] =  d['Close_'+t] - d['Low_'+t]
            out[f"co_1_{t}"] =  d['Close_'+t] - d['Open_'+t]
            out[f"ho_1_{t}"] =  d['High_'+t] - d['Open_'+t]
            out[f"ol_1_{t}"] =  d['Open_'+t] - d['Low_'+t]
            out[f"gap_1_{t}"]= d['Open_'+t] - d['Close_'+t].shift(1)
            out[f"bop_{t}"] = getattr(talib, 'BOP')(d['Open_'+t], d['High_'+t],d['Low_'+t],d['Close_'+t])
            
            for l in [1] + n:
                out[f"r_{str(l)}_{t}"] = np.log(d['Close_'+t] / d['Close_'+t].shift(l))    # log return
                out[f"v_{str(l)}_{t}"] = d['Volume_'+t].rolling(l).mean()
                if l>1:
                    if t == self.inst.ticker:
                        if self.vix is not None:
                            out[f"std_vx_ratio{str(l)}_{t}"] = out[f"r_1_{t}"].rolling(l).std()*math.sqrt(bars_per_day*252)*100./out[f"Close_VX"]
                    out[f"std_{str(l)}_{t}"] = out[f"r_1_{t}"].rolling(l).std()*math.sqrt(bars_per_day*252)
                    out[f"d_{str(l)}_{t}"] = out[f"d_1_{t}"].rolling(l).mean()
                    out[f"max_{str(l)}_{t}"] = d[f"High_{t}"].rolling(l).max() - d['Close_'+t]
                    out[f"min_{str(l)}_{t}"] = d['Close_'+t] - d[f"Low_{t}"].rolling(l).min()
                    out[f"range_{str(l)}_{t}"] = out[f"hl_1_{t}"].rolling(l).mean()
                    out[f"hc_{str(l)}_{t}"] = out[f"hc_1_{t}"].rolling(l).mean()
                    out[f"cl_{str(l)}_{t}"] = out[f"cl_1_{t}"].rolling(l).mean()
                    out[f"gap_{str(l)}_{t}"] = out[f"gap_1_{t}"].rolling(l).mean()
                    out[f"co_{str(l)}_{t}"] =   out[f"co_1_{t}"].rolling(l).mean()
                    out[f"dsma_{str(l)}_{t}"] = d['Close_'+t] - d[f"Close_{t}"].rolling(l).mean()
                if l>2:
                    out[f"adx_{str(l)}_{t}"] = getattr(talib, 'ADX')(d['High_'+t], d['Low_'+t], d['Close_'+t],timeperiod=l) # ADX 
                    out[f"apo_{str(l)}_{t}"] = getattr(talib, 'APO')(d['Close_'+t],fastperiod=l,slowperiod=l*2) # Absolute Price Oscillator 
                    out[f"aronosc_{str(l)}_{t}"] = getattr(talib, 'AROONOSC')(d['High_'+t], d['Low_'+t],timeperiod=l) # Aron Oscillator 
                    out[f"cci_{str(l)}_{t}"] = getattr(talib, 'CCI')(d['High_'+t], d['Low_'+t], d['Close_'+t],timeperiod=l) # CCI
                    out[f"mfi_{str(l)}_{t}"] = getattr(talib, 'MFI')(d['High_'+t], d['Low_'+t], d['Close_'+t],d['Volume_'+t],timeperiod=l) #  Money Flow Index
                    out[f"rsi_{str(l)}_{t}"] = getattr(talib, 'RSI')(d['Close_'+t],timeperiod=l) # Absolute Price Oscillator
                    out[f"obv_{str(l)}_{t}"] = self.RollingOBV(d, t, l) # rolling OBV
                    out[f"natr_{str(l)}_{t}"] = getattr(talib, 'NATR')(d['High_'+t], d['Low_'+t], d['Close_'+t],timeperiod=l) # normalised ATR

        
        #Seasonality and micro seasonality    
        out['DayOfWeek'] = d.index.day_of_week
        if self.freq is not '1d': 
            out['Hour']  = d.index.hour
        
        out.dropna(inplace=True)
        self.number_of_features = len(out.columns)
        self.features_data = out

    def create_label(self,target_up=0.25,target_down=-0.25,n=1):
        """
        Given a DataFrame d with OHLCV price data, it returns a time series with the following labels:
            long_signal:
                1: The Close of the next n bars is greater or equal than the next bar Open + target_up 
                0: Otherwise
            short_signal:
                1: The Close of the next n bars is less  than the next bar Open + target_down
                0: Otherwise
        -------------------------------------------------------------------------------------
        -------------------------------------------------------------------------------------
        
        Inputs:
        -------
        target_up:  float
            target for long signal
        target_down: float
            target for short signal
        n: int
            number of bars ahead
        """
        
        d = self.cleansed_data
        
        out = pn.DataFrame(index=d.index)
        
        t = '_' + self.inst.ticker
        
        future_close = d['Close'+t].shift(-n)
        
        out['long_label'] = np.where(future_close - d['Open'+t].shift(-1) > target_up ,1,0)
        out['short_label'] = np.where(future_close - d['Open'+t].shift(-1) <  target_down ,1,0)
        
        out.dropna(inplace=True)
        
        self.labels_data = out
        self.max_bars_per_day = self.number_of_bars_per_day(out)
        self.n = n
      
    def standardise(self,train_,test_,clip=None):
        """
        Standardise all columns of training and test DataFrames for ML predictions. 
        Test standardisation is based on mean and standard deviation of thr training data 
        -------------------------------------------------------------------------------------
        -------------------------------------------------------------------------------------
        
        Inputs:
        -------
        train: DataFrame
            DataFrame containing the ML predictor
        test: DataFrame  
            DataFrame with the same columns of train used to test the prediction
        clip: float or None
            if not None, standardised values are clipped at +- the clip value
            
        """ 
        
        out_train = pn.DataFrame(index=train_.index)
        out_test = pn.DataFrame(index=test_.index)
        
        for column in train_:
            mu, std = train_[column].mean(), train_[column].std()
            out_train[column] = (train_[column] - mu) / std
            out_test[column] = (test_[column] - mu) / std
        if (clip is not None):
            out_train = out_train.clip(-clip,clip)
            out_test = out_test.clip(-clip,clip)
        
        return out_train, out_test
    
    
    def split_data_standardise(self, dates, clip):
        '''
        split features and labels into training and test and standardises features
        
        Inputs:
        ----------------------
        
        dates: list of DateTimes [date1,date2,date3]
                training > date1 and <=date2 , test > date2 and <= date3
        
        clip: float or None
            if not None, standardised values are clipped at +- the clip value
            
        '''
        
        features = self.features_data
        labels   = self.labels_data
        
        self.features_train = features.loc[(features.index > dates[0]) & (features.index <= dates[1])]
        self.labels_train = labels.loc[(labels.index > dates[0]) & (labels.index <= dates[1])]
        self.features_test = features.loc[(features.index > dates[1]) & (features.index <= dates[2])]
        self.labels_test = labels.loc[(labels.index > dates[1]) & (labels.index <= dates[2])]
                    
        #standardise features
        self.features_train, self.features_test = self.standardise(self.features_train, self.features_test,
                                                                  clip)        
        #align features and lables dataframes
        self.features_train, self.labels_train = self.features_train.align(self.labels_train, join='inner',axis=0)
        self.features_test, self.labels_test = self.features_test.align(self.labels_test, join='inner',axis=0)
        
    def set_trading_hours(self,start_time="00:00:00",end_time="23:59:59"):
        '''
        remove bars outside before start_time and after end_time so that the ML model 
        is calibrated on the same time range of the trading strategy
        
        Inputs:
        ----------------------
        start_time: str: HH:MM:SS
        end_time  : str: HH:MM:SS
        
        '''
        
        start_time = datetime.strptime(start_time,"%H:%M:%S").time()
        end_time = datetime.strptime(end_time,"%H:%M:%S").time()
        
        self.features_train = self.features_train[(self.features_train.index.time >= start_time) & (self.features_train.index.time < end_time)]
        self.labels_train = self.labels_train[(self.labels_train.index.time >= start_time) & (self.labels_train.index.time < end_time)]
        self.features_test = self.features_test[(self.features_test.index.time >= start_time) & (self.features_test.index.time < end_time)]
        self.labels_test = self.labels_test[(self.labels_test.index.time >= start_time) & (self.labels_test.index.time < end_time)]
        
        
    def features_scatter(self,cols):
        '''
        display multivariate scatter plot of selected features
        
        Inputs:
        ----------------------
        cols: list
             features to be displayed
        '''
        
        g = sns.PairGrid(self.features_train,vars=cols)
        g.map_diag(sns.histplot)
        g.map_offdiag(sns.scatterplot)
    
    def joint_plot(self,x1,x2,long=True):
        '''
        display joint plot of two selected features (x1,x2) against positive and negative label distributions
        
        Inputs:
        ----------------------
        x1: str
             name of first feature
        x2: str 
            name of second feature
        long: bool
            long label?
        '''
        
        label_type = 'long_label' if long==True else 'short_label'
        pos_df = pn.DataFrame(self.features_train.loc[self.labels_train[label_type]>=0.5], columns=self.features_train.columns)
        neg_df = pn.DataFrame(self.features_train.loc[self.labels_train[label_type]<0.5], columns=self.features_train.columns)
        
        sns.jointplot(x= pos_df[x1], y =pos_df[x2],
                          kind='hex')
        plt.suptitle("Positive distribution")
        
        sns.jointplot(x =neg_df[x1], y = neg_df[x2],
                          kind='hex')
        _ = plt.suptitle("Negative distribution")
    

class Predictor(object):
        '''
        Base class for a ML Predictor object that contains main parameters and methods shared across
        different ML techniques
        
        Attributes:
            Inputs
            --------------------
            optimiser: str
                'adam' or 'sgd'
            ephocs: int
                number of ephocs
            long: bool
                if true, it predicts long label - otherwise the short label is used
            early_monitor: str
                performance metric for early stopping
            strategy_data: strategy data class
                cleansed data for model calibration and back-testing
            selected_features: list containing column names to be used for model calibration.If None, all the features in the
                strategy data class are used
                
                
            Data:
            ----------------------------------
            label_bars_ahead: int 
                number of bars ahead used to construct label
            boruta_selector: obj
                calibrated boruta object
            selected_features: list 
                    list containing column names to be used for a model calibration based on a more parsimonious number 
                    of features. The list can be an input of the class or the output of the variable selection method. 
                    If None, all generated features are used for model calibration
        
        Methods:
            set_metrics:
                defines performance metrics of ML algorithm
            set_optimiser:
                sets optimiser (adam or SGD)
            cw:
                calculates weights for label class imbalance
            set_seeds:
                set seeds
            set_early_stopping:
                set early stopping settings of calibration algorithm
            choose_label:
                returns long or short label for ML fit 
            plot_metrics_history:
                plot performance metrics across ephocs
            plot_cm:
                plot confusion matrices
            plot_prc:
                plot precision recall curve
            plot_roc:
                plot roc curve
            plot_metrics:
                plot confusion matrices, prc and roc
            print_model
                print model structure
            calculate_gini
                calculate gini for test and training
            boruita_fit
                fit boruta method for feature selection - store result in prperty boruta_selector
            features_selection
                generate a list of slected features based on Boruta approach
                
        '''
        
        def __init__(self,optimiser,epochs,long,early_monitor,strategy_data,selected_features):
            self.set_optimiser(optimiser)
            self.set_metrics()
            self.set_early_stopping(early_monitor)
            self.set_seeds()
            self.model = None
            self.epochs = epochs
            self.long = long
            self.long_str = 'Long Label' if long else 'Short Label'
            self.label_bars_ahead = strategy_data.n
            self.selected_features = strategy_data.features_train.columns if selected_features is None else selected_features

            
        def set_metrics(self):
            '''
            Defines Performance metrics of ML calibration
            '''
            self.METRICS = [
            keras.metrics.TruePositives(name='tp'),
            keras.metrics.FalsePositives(name='fp'),
            keras.metrics.TrueNegatives(name='tn'),
            keras.metrics.FalseNegatives(name='fn'), 
            keras.metrics.BinaryAccuracy(name='accuracy'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc'),
            keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
            ]
        
        def set_optimiser(self, name, lr=0.001, momentum=0., nesterov=False):
            '''
            set optimiser
            '''
            if name == 'adam': 
                self.optimiser = Adam(lr = lr)
            elif name == 'sgd':
                self.optimiser = SGD(lr=lr, momentum=momentum, nesterov=nesterov)
            else:
                self.optimiser = name
        
        def cw(self,ts):
            '''
            Calculate weights for label's class imbalance
            '''
            c0, c1 = np.bincount(ts)
            w0 = (1 / c0) * (len(ts)) / 2
            w1 = (1 / c1) * (len(ts)) / 2
            return {0: w0, 1: w1}
            
        def set_seeds(self,seed=100):
            '''
            set random seeds to generate the same results
            '''
            random.seed(seed)  
            np.random.seed(seed)  
            tf.random.set_seed(seed)
        
        def set_early_stopping(self,monitor):
            self.early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor=monitor, 
            verbose=1,
            patience=40,
            mode='min',
            restore_best_weights =False)
            
        def choose_label(self,strategy_data, long=True):
            '''
            return long (or short) training and test labels
            
            inputs:
                strategy_data: strategy_data class
                long:bool
                    if true, the long label is returned - false for short
            '''
            if long == True:
                label = strategy_data.labels_train['long_label']
                label_test = strategy_data.labels_test['long_label']
            else:
                label = strategy_data.labels_train['short_label']
                label_test = strategy_data.labels_test['short_label']
            
            return label, label_test
            
        def plot_metrics_history(self,history):
            '''
            plot metrics by ephocs
            '''
            out = pn.DataFrame(history)
            metrics = ['loss', 'accuracy', 'precision', 'recall']
            plt.figure(figsize=(12,8))
            for n, metric in enumerate(metrics):
                name = metric.replace("_"," ").capitalize()
                plt.subplot(2,2,n+1)
                plt.title(str(metric)+'_'+ self.long_str)
                plt.plot(out.index, out[metric], color='blue', label='Train')
                plt.plot(out.index, out['val_'+metric],
                        color='green', linestyle="--", label='Val')
                plt.xlabel('Epoch')
                plt.ylabel(name)
                if metric == 'loss':
                    plt.ylim([0, plt.ylim()[1]])
                elif metric == 'auc':
                    plt.ylim([0.5,1])
                else:
                    plt.ylim([0,1])
                plt.legend()  
        
        def plot_cm(self,labels, predictions,name='Training', p=0.5):
            '''
            plot confusion matrix
            '''
            cm = confusion_matrix(labels, predictions > p)
            cm = cm / np.sum(cm)
            plt.figure(figsize=(5,5))
            sns.heatmap(cm, annot=True, 
                fmt='.2%', cmap='Blues')
            plt.title(self.long_str + ' ' + name + ' Confusion matrix @{:.2f}'.format(p))
            plt.ylabel('Actual label')
            plt.xlabel('Predicted label')
        
        def plot_prc(self,name, labels, predictions, **kwargs):
            '''
            plot precision recall curve
            '''
            precision, recall, _ = sklearn.metrics.precision_recall_curve(labels, predictions)
            plt.plot(precision, recall, label=name, linewidth=2, **kwargs)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.grid(True)
            ax = plt.gca()
            ax.set_aspect('equal')
        
        def plot_roc(self,name, labels, predictions, **kwargs):
            '''
            plot roc curve
            '''
            fp, tp, _ = sklearn.metrics.roc_curve(labels, predictions)
            print('GINI '+name +': ' + str(sklearn.metrics.auc(fp, tp)*2.-1.))
            plt.plot(100*fp, 100*tp, label=name, linewidth=2, **kwargs)
            plt.xlabel('False positives [%]')
            plt.ylabel('True positives [%]')
            plt.grid(True)
            ax = plt.gca()
            ax.set_aspect('equal')
        
        def plot_metrics(self,train_labels,train_predictions,test_labels,test_predictions,p_=0.5):
            '''
            plot confusion matrices, precision recall curve, roc curve 
            '''
            self.plot_cm(train_labels,train_predictions,p=p_)
            self.plot_cm(test_labels,test_predictions,name='Test',p=p_)
        
            plt.figure(figsize=(12,8))
            plt.subplot(1,2,1)
            self.plot_prc(self.long_str + " Train", train_labels, train_predictions, color='blue')
            self.plot_prc(self.long_str + " Test", test_labels, test_predictions, color='green', linestyle='--')
            plt.legend(loc='lower right')
            plt.title(self.long_str + ' Precision-Recall Curve')
            plt.subplot(1,2,2)
            self.plot_roc(self.long_str + " Train", train_labels, train_predictions, color='blue')
            self.plot_roc(self.long_str + " Test", test_labels, test_predictions, color='green', linestyle='--')
            plt.legend(loc='lower right')
            plt.title(self.long_str + ' ROC Curve')
    
        def calculate_confusion_matrix(self,labels,predictions, p=0.5):
            
            if confusion_matrix(labels, predictions > p).ravel().size == 4:
                tn, fp, fn, tp = confusion_matrix(labels, predictions > p).ravel()
                accuracy = (tp + tn) / (tn + fp + fn + tp)
                
                if (tp + fp) > 0:
                    precision = tp / (tp + fp)
                else:
                    precision = None
                
                if (tp + fn) > 0:
                    recall = tp / (tp + fn)
                else:
                    recall = None
            else:
                    accuracy = None
                    precision = None
                    recall = None
            
            return accuracy,precision,recall
        
        def calculate_gini(self,train_labels,train_predictions,test_labels,test_predictions):
            '''
            calculate gini of train and test
            '''
            fp, tp, _ = sklearn.metrics.roc_curve(train_labels, train_predictions)
            self.train_gini = sklearn.metrics.auc(fp, tp)*2.-1.   
            
            self.train_accuracy, self.train_precision, self.train_recall = self.calculate_confusion_matrix(train_labels,
                                                                                                        train_predictions)
            fp, tp, _ = sklearn.metrics.roc_curve(test_labels, test_predictions)
            self.test_gini = sklearn.metrics.auc(fp, tp)*2.-1.  
            
            self.test_accuracy, self.test_precision, self.test_recall = self.calculate_confusion_matrix(test_labels,
                                                                                                        test_predictions)
        
        def print_model(self):
            '''
            print model summary
            '''
            print(self.model.summary())
        
        def boruta_fit(self,strategy_data):
            '''
            Apply Boruta approach for variable selection  
            
            inputs:
                strategy_data: strategy_data class
            
            '''
            
            y,_ = self.choose_label(strategy_data, long = self.long)
            rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5, random_state=42)
            boruta_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=42)
            boruta_selector.fit(strategy_data.features_train.values,y)
            self.boruta_selector = boruta_selector
        
        def boruta_features_selection(self,strategy_data,ranking_thr = None):
            '''
            save selected features based on boruta method
            
            inputs:
                strategy_data: strategy_data class        
                ranking_thr: int 
                    select features that rank higher than ranking_thr. If None, use Boruta selected features 
            '''
                
            if ranking_thr == None:
                self.selected_features = strategy_data.features_train.columns[self.boruta_selector.support_]  
            else:
                self.selected_features = strategy_data.features_train.columns[self.boruta_selector.ranking_<=ranking_thr]
            
        def drop_correlated_features(self,strategy_data,corr_threshold = 0.7):
            '''
            remove from selected features list the features with correlation higher than threshold
            
            inputs:
                strategy_data: strategy_data class        
                corr_threshold: float 
                    correlation threshold 
            '''
            
            c = strategy_data.features_train[self.selected_features].corr()

            # Identify highly correlated features
            correlated_features = set()
            for i in range(len(c.columns)):
                for j in range(i):
                    if abs(c.iloc[i, j]) > corr_threshold:
                        colname = c.columns[i]
                        correlated_features.add(colname)
            # remove highly correlated features
            self.selected_features = self.selected_features.drop(correlated_features)
            
        
class DNN(Predictor):
    '''
    Dense Neural Network class
    
    Attributes:
        Inputs
        --------------------
        strategy_data: strategy data class
            class containing the strategy data
        optimiser: str
            'adam' or 'sgd'
        hl: int
            number of hidden layers
        hu: int
            number of hidden units
        dropout: bool
            if true, apply random dropout of nodes
        rate: float
            dropout rate if dropout is true
        regularize: bool
            regularize nodes
        reg_l1=float
            regularisation l2 parameter
        ephocs: int
            number of ephocs
        long: bool
            if true - long label is used
        early_monitor: str
            performance metric for early stopping   
            
        Data:
        ----------------------------------------
        model: sequential object
            tensor flow model
        results: 
            model results
        train_predict:
            training prediction probability
        test_predict:
            test prediction probability
        train_label:
            training labels
        test_label:
            test_labels
        train_gini:
            gini of training
        test_gini:
            gini of test
        
        Methods:
        -----------------------------------------
        create model
        fit
        predict
    
    
    '''
    
    def __init__(self,strategy_data,optimiser='adam',hl=1,hu=128,dropout=False,rate=0.3,regularize=False,
                reg_l1=0.0005,epochs=100,long=True,early_monitor='val_loss', selected_features = None):
        super().__init__(optimiser = optimiser, epochs = epochs, long = long, 
                         early_monitor = early_monitor, strategy_data = strategy_data, 
                         selected_features = selected_features)
        self.strategy_data = strategy_data
        self.hl = hl
        self.hu = hu
        self.dropout = dropout
        self.rate = rate
        self.regularize = regularize
        self.reg_l1=reg_l1
        self.train_label,self.test_label = self.choose_label(strategy_data,self.long)
        
    def create_model(self,hl, hu, dropout, rate, regularize, reg, optimiser, 
                     input_dim, metrics):
        '''
        create NN model
        '''
        
        if not regularize:
            reg = None
        model = Sequential()
        model.add(Dense(hu, input_dim=input_dim,
                        activity_regularizer=reg,  
                        activation='relu'))
        if dropout:
            model.add(Dropout(rate, seed=100))
        for _ in range(hl):
            model.add(Dense(hu, activation='relu',
                            activity_regularizer=reg))  
            if dropout:
                model.add(Dropout(rate, seed=100))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer=optimiser,
                      metrics= metrics)
        self.model = model
    
    
    def fit(self,strategy_data):
        '''
        fit model
        '''
        
        self.create_model(self.hl,self.hu,self.dropout,self.rate,self.regularize,
                          l1(self.reg_l1),self.optimiser,len(self.selected_features),self.METRICS)
        
        self.results =  self.model.fit(strategy_data.features_train[self.selected_features],
                                        self.train_label,epochs=self.epochs, 
                                        verbose=False,validation_split=0.3, shuffle=False,
                                        class_weight=self.cw(self.train_label),
                                        callbacks=[self.early_stopping])
            
            
    def predict(self,strategy_data,show_charts=False):
        '''
        predict training and out of sample
        '''
        self.train_predict = pn.Series(self.model.predict(strategy_data.features_train[self.selected_features]).flatten(), 
                                       index = self.train_label.index)
        
        self.test_predict = pn.Series(self.model.predict(strategy_data.features_test[self.selected_features]).flatten(), 
                                      index = self.test_label.index)
        
        self.calculate_gini(self.train_label,self.train_predict,self.test_label,self.test_predict)
        
        if show_charts:
            self.plot_metrics_history(self.results.history)
            self.plot_metrics(self.train_label,self.train_predict,self.test_label,self.test_predict)
            


class DRNN(Predictor):
    '''
    Dense Recurrent Neural Network class
    
    Attributes:
        Inputs
        --------------------
        strategy_data: strategy data class
            class containing the strategy data
        lags: int
            number of lags
        optimiser: str
            'adam' or 'sgd'
        hl: int
            number of hidden layers
        hu: int
            number of hidden units
        layer: str
            "SimpleRNN" or "LTSM"
        dropout: bool
            if trueapply random dropout of nodes
        rate: float
            dropout rate if dropout is true
        ephocs: int
            number of ephocs
        long: bool
            if true - long label is used
        early_monitor: str
            performance metric for early stopping   
        
        Data:
        ----------------------------------------
        model: sequential object
            tensor flow model
        results: 
            model results
        train_predict:
            training prediction probabilities
        test_predict:
            test prediction probabilities
        train_label:
            training labels
        test_label:
            test_labels
            
        Methods:
        -----------------------------------------
        create model
        fit
        predict

    '''
    def __init__(self,strategy_data,lags=5,optimiser='rmsprop',hl=1,hu=100,layer='SimpleRNN',
                dropout=False,rate=0.3,epochs=50,long=True, early_monitor='loss',selected_features = None):
        super().__init__(optimiser = optimiser, epochs = epochs, long = long, 
                         early_monitor = early_monitor, strategy_data = strategy_data,
                         selected_features = selected_features)
        self.hl = hl
        self.hu = hu
        self.layer = layer
        self.dropout= dropout
        self.rate = rate
        self.optimiser = optimiser
        self.lags = lags
        self.train_label,self.test_label = self.choose_label(strategy_data,self.long)
    
    
    def plot_metrics_history(self,history):
        '''
        plot metrics by ephocs
        '''
        out = pn.DataFrame(history)
        metrics = ['loss', 'accuracy', 'precision', 'recall']
        plt.figure(figsize=(12,8))
        for n, metric in enumerate(metrics):
            name = metric.replace("_"," ").capitalize()
            plt.subplot(2,2,n+1)
            plt.plot(out.index, out[metric], color='blue', label='Train')
            plt.xlabel('Epoch')
            plt.ylabel(name)
            if metric == 'loss':
                plt.ylim([0, plt.ylim()[1]])
            elif metric == 'auc':
                plt.ylim([0.5,1])
            else:
                plt.ylim([0,1])
            plt.legend() 
    
    
    def create_model(self,hl, hu, layer,optimizer, features,dropout, rate,lags_,metrics):
        '''
        create DRNN model
        '''
        if hl <= 2: 
                hl = 2  
        if layer == 'SimpleRNN':
            layer = SimpleRNN
        else:
            layer = LSTM
        model = Sequential()
        model.add(layer(hu, input_shape=(lags_, features),
                                   return_sequences=True))  
        if dropout:
            model.add(Dropout(rate, seed=100))  
        for _ in range(2, hl):
            model.add(layer(hu, return_sequences=True))
            if dropout:
                model.add(Dropout(rate, seed=100))  
        model.add(layer(hu))  
        model.add(Dense(1, activation='sigmoid'))  
        model.compile(optimizer=optimizer,
                          loss='binary_crossentropy',
                          metrics=metrics)
        self.model = model
        
    def fit(self,strategy_data,batch_size = 5,steps_per_epoch=10):
        
        self.create_model(self.hl,self.hu,self.layer,self.optimiser,len(self.selected_features),
                          self.dropout, self.rate,self.lags,self.METRICS)
        
        self.g = TimeseriesGenerator(strategy_data.features_train[self.selected_features], 
                                self.train_label, length=self.lags, batch_size=batch_size)
        
        
        
        self.results = self.model.fit(self.g, epochs=self.epochs, steps_per_epoch=steps_per_epoch,
                                      verbose=False, class_weight=self.cw(self.train_label),
                                      callbacks=[self.early_stopping])
        

    
    def predict(self,strategy_data,batch_size = 5,show_charts=False):
        '''
        predict training and outof sample
        '''
        
        self.train_predict = pn.Series(self.model.predict(self.g, batch_size=None).flatten(),
                                        index = self.train_label.index[self.lags:])
        
        g_test = TimeseriesGenerator(strategy_data.features_test[self.selected_features], 
                                         self.test_label,length=self.lags, batch_size=batch_size)
        
        self.test_predict = pn.Series(self.model.predict(g_test, batch_size=None).flatten(),
                                        index = self.test_label.index[self.lags:])
        
        self.train_label = self.train_label[self.lags:]
        self.test_label = self.test_label[self.lags:]
        
        self.calculate_gini(self.train_label,self.train_predict,self.test_label,self.test_predict)
        
        if show_charts:
            self.plot_metrics_history(self.results.history)
            self.plot_metrics(self.train_label,self.train_predict,self.test_label,self.test_predict)

class Strategy(object):
    '''
    Strategy object for vectorized backtesting
    
    Attributes:
        Inputs
        --------------------
        strategy_data: strategy_data class 
            cleansed data for ML prediction and back-testing
        predictor_long: predictor class
            calibrated long ML
        predictor_short
            calibrated short ML object
        p_thr: float
            probability threshold for long and short signal
        max_w:float
            maximum long and short weight (ex. if equal to 1 then -1<=w<=1)
        point_value: float
            point value of instrument to be back-tested
        commissions: float
            commissions per trade
        slippage: float
            slippage per trade
        
        
        Data:
        ----------------------------------------
        w: DataFrame
            strategy weights ('long','short','long_short','benchmark_long')
        gross_perf_metrics: Dataframe
        gross performance metrics: Dataframe
        gross_cumulative_equity: Dataframe
        gross_log_returns: Dataframe
        net_log_return: Dataframe
        net_cumulative_equity: Dataframe
        net_perf_metrics: Dataframe
        
            
        Methods:
        -----------------------------------------
        compute_weights:
            calculates strategy weights
        calculate_perf_metrics
        run_backtest
        
        

    '''
    def __init__(self, strategy_data, predictor_long, predictor_short,p_thr=0.5, max_w=1.,
                point_value=50,commission=2.,slippage=0.25,show_chart=False):
        self.max_bars_per_day = strategy_data.max_bars_per_day
        self.long_bars_ahead = predictor_long.label_bars_ahead 
        self.short_bars_ahead = predictor_short.label_bars_ahead
        self.max_w = max_w
        
        
        self.compute_weights(predictor_long.test_predict, predictor_short.test_predict,
                            predictor_long.label_bars_ahead, predictor_short.label_bars_ahead)
        
        
        self.ticker =  strategy_data.inst.ticker 
        self.run_backtest(strategy_data.cleansed_data['Close_'+self.ticker],
                          self.w,point_value=point_value,
                          commission=commission,slippage=slippage,show_chart=show_chart)
        
    
    def compute_weights(self,pred_long, pred_short,bars_long,bars_short,p_thr=0.5):
        '''
        Calculates strategy weights (long only, short only and long+short) based probabilities
        of long (and short) label within the next "bars_long" (and "bars_short") bars
        
        is "pred_long probability" at bar (i) is higher than "p_thr" 1. is added to the weights 
        of the next n ("bars_long"). weights are then capped and flored to range between -1 and 1
        
        
        Inputs:
        --------------------------------------------------
            pred_long: Series
                long predicted probabilities
            pred_short: Series
                short predicted probabilities
            bars_long: int
                bars ahead for long prediction 
            bars_short: int
                bars ahead for short prediction 
            p_thr: float
                probability threshold for long signal
        
        '''

        w = pn.DataFrame(index=pred_long.index.union(pred_short.index))

        w['long'] = 0.    # long only strategy
        w['short'] = 0.   # short only strategy
        for i in range(1,bars_long+1):
            w['long'] = w['long'] + np.where((pred_long.shift(i)>p_thr),1.,0.)
        for i in range(1,bars_short+1):
            w['short'] = w['short'] + np.where((pred_short.shift(i)>p_thr),-1.,0.)
        w['long_short'] = w['long'] + w['short']  # long short strategy

        w = w.clip(-1.*self.max_w,self.max_w)
        
        #passive strategies to be used as benchmarks
        w['baseline_long'] = 1.
        
        self.w = w
    
    def calculate_perf_metrics(self,p,num_bars_per_day):
        '''
        It returns performance metrics (average return, average standard deviation, Information Ratio) 
        from an array of strategy returns
        
        Inputs:
        -------------------------
        p: DataFrame
            dataframe with multiple strategies' returns
        num_bars_per_day: int
            number of bars per day. It is used to annualise the performance metrics 
        
        '''
        perf_stats = pn.DataFrame()
        perf_stats['mean_return']=p.mean()*num_bars_per_day*252
        perf_stats['std_return']=p.std()*math.sqrt(num_bars_per_day*252)
        perf_stats['IR']= perf_stats['mean_return'] / perf_stats['std_return']
        return perf_stats
    
    def run_backtest(self,c,w,point_value=50,commission=2.,slippage=0.25,show_chart=False):
        '''
        backtest trading strategies given weights, capital and transactions costs 
        
        Inputs:
        ----------------------------
            c: Series
                Close time series of the instrument that is traded 
            w: Dataframw
                weights a various strategies
            initial_capital: float
            point_value: float
                value of 1 unit of the traded instrument
            commission: float
            slippage: float

        '''

        gross_perf = pn.DataFrame(index = c.index)
        
        log_ret = np.log(c / c.shift(1)) 
        
        gross_perf['long'] = w['long'] * log_ret
        gross_perf['short'] = w['short'] * log_ret
        gross_perf['long_short'] = w['long_short'] * log_ret
        gross_perf['baseline_long'] = log_ret
        gross_perf.dropna(inplace=True)
        
        nn = max(self.long_bars_ahead ,self.short_bars_ahead )
        gross_perf = gross_perf.iloc[nn:,:]

        self.gross_perf_metrics = self.calculate_perf_metrics(gross_perf,self.max_bars_per_day).round(2)
        self.gross_log_returns = gross_perf 
        self.gross_cumulative_equity = gross_perf.cumsum().apply(np.exp)
        
        if show_chart == True:
            self.gross_cumulative_equity.plot(figsize=(16, 6),
                                                   title='gross strategy returns');
            print('*****Gross Performance*****')
            print('')
            print(self.gross_perf_metrics)         
        
        perf = pn.DataFrame(index = c.index)
        
        avg_p = c[gross_perf.index].mean()
        ptr =  (commission/point_value + slippage)/avg_p #proportional transaction cost 
        self.cost = (w != w.shift(1))* 1. * ptr
        
        perf['long'] = w['long'] * log_ret -self.cost['long']
        perf['short'] = w['short'] * log_ret -self.cost['short']
        perf['long_short'] = w['long_short'] * log_ret - self.cost['long_short']
        perf['baseline_long'] = log_ret
        perf = perf.iloc[nn:,:]
        perf.dropna(inplace=True)
        
        self.Net_log_returns = perf 
        self.net_cumulative_equity =  perf.cumsum().apply(np.exp)
        self.net_perf_metrics = self.calculate_perf_metrics(perf,self.max_bars_per_day).round(2)
        
        if show_chart == True:
            self.net_cumulative_equity.plot(figsize=(16, 6),title='net strategy cumulative returns');
            print('*****Net Performance*****')
            print('')
            print(self.net_perf_metrics)
    

class cross_validation(object):
        '''
        this class is used to run the full process from data preparation, ML calibration, prediction and backtetsing. 
        It can be used to run a single or multiple combinations of inputs. Input and output performance 
        metrics are stored in the "input_output_df" dataframe. Performance charts are also generated.
        
        Attributes:
            Data:
            -----------------------------------------------
            combo_id: int
                id of combination of inputs
            input_output_df: DataFrame
                each row is a combination of inputs to be tested. the columns contain 
                input and output metrics
                
        Methods:
            initialise_input_df
            add_inputs_combo
                add combination of inputs to inputs_dataframe
            run

            '''
        def __init__(self):
            self.initialise_input_df()
            
        def initialise_input_df(self):
            '''
            create the inputs dataframe
            
            list of inputs:
            StrategyData
                frequency: str 
                    frequency of data
                vix: bool
                    include vix
                target_label: float
                    minimum up or down change in price (based on number of standard deviations) for label to be true
                n label: int
                    bars ahead for label
                clip: float
                    cap and floor of standardised variables
                date1: datetime
                    start of training sample 
                date2: datetime
                    end of training and start of test
                date3: datetime
                    end of test
                start_trading_time: time
                end_trading_date: time
                run_boruta: bool
                    reduce number of features based on boruta
                drop_correlated_features: bool
                    drop correlated features
                ephocs: int
                    number of ephocs
                long: bool 
                    if true, long signal
                ml_type: str
                    'DNN' or 'RDNN'
                optimiser: str
                    optimiser name
                lr: float
                    lr parameter for optimiser
                momentum: float
                    momentum for SGD optimiser
                nesterow: bool 
                    parameter for SGD optimiser
                hl: int
                    hidden layers
                hu: int
                    hidden units
                dropout: bool
                    dropout true false
                rate: float
                    dropout rate
                regularize: bool
                    regularise units
                reg_l1:
                    l1 regularisation parameter
                lags: int
                    number of lags for RDNN
                layer:str
                    RDNN layer type: "SimpleRNN" or "LTSM"

            '''        
            param_names = ['frequency','vix','target_label','n_label','clip','date1','date2','date3','start_trading_time',
                        'end_trading_time','run_boruta','drop_correlated_features','epochs','long','ml_type',
                        'optimiser','lr','momentum','nesterov','hl','hu','dropout','rate','regularize','reg_l1',
                        'lags','layer']
            self.input_output_df = pn.DataFrame(columns=param_names)
            self.combo_id=0
            
    
        def to_csv(self,name):
            self.input_output_df.to_csv(name +'.csv')
        
        def add_input_combo(self,frequency='30min',vix=True,target_label=1.,n_label=1,clip=5.,
                                date1 = '2015-12-31 23:59:00',date2 = '2017-12-31 23:59:00' ,
                                date3 = '2018-12-31 23:59:00',start_trading_time = "00:08:00",
                                end_trading_time = "17:59:59", run_boruta = False,
                                drop_correlated_features = False, epochs = 50,long = True, ml_type = 'DNN',
                                optimiser = 'adam',lr = 0.001, momentum = 0.8 ,nesterov=True,hl = 1,hu=16,
                                dropout = True, rate = 0.3,regularize = True, reg_l1 = 0.0005 ,
                                lags = 5,layer = 'SimpleRNN'):
            '''
            add new combination of parameters to "input_output_df" dataframe. the run method will calibrate and test the 
            provided combination
            '''
            row = [frequency,vix,target_label,n_label,clip,date1,date2,date3,start_trading_time, end_trading_time,
                        run_boruta, drop_correlated_features, epochs,long,ml_type,
                        optimiser,lr,momentum,nesterov,hl,hu,dropout,rate,regularize,reg_l1,lags,layer]          
            self.input_output_df.loc[self.combo_id] = row
            self.combo_id = self.combo_id + 1 
    
        def add_multiple_input_combos(self,frequency=['30min'],vix=[True],target_label=[1.],
                                n_label=[1],clip=[5.],
                                date1 = ['2015-12-31 23:59:00'],date2 = ['2017-12-31 23:59:00'] ,
                                date3 = ['2018-12-31 23:59:00'],start_trading_time = ["00:08:00"],
                                end_trading_time = ["17:59:59"],run_boruta = [False],
                                drop_correlated_features = [False], epochs = [50],long = [True], ml_type = ['DNN'],
                                optimiser = ['adam'],lr = [0.001], momentum = [0.8] ,nesterov=[True],hl = [1],
                                hu=[16], dropout = [True], rate = [0.3],regularize = [True], reg_l1 = [0.0005] ,
                                lags = [5],layer = ['SimpleRNN']):
            '''
            add multiple combinations of inputs to the input_output_df dataframe. the run_multiple method will iterate across
            the rows of the inputs dataframe to calibrate and test all of them.
            
            multiple values can be added to each field as list elements. A cartesian product across all lists of inputs is
            done to generate all the combinations

            '''
            
            param_names = [frequency,vix,target_label,n_label,clip,date1,date2,date3,start_trading_time, 
                        end_trading_time ,run_boruta, drop_correlated_features, 
                        epochs,long,ml_type,optimiser,lr,momentum,nesterov,hl,hu,dropout,rate,
                        regularize,reg_l1,lags,layer]
            
            combos = list(itertools.product(*param_names))
            
            for x in combos:
                self.add_input_combo(*x)

                
        
        
        def save_output(self, combo_id, data, model, backtest):
            '''
            save output of ML algorithm  in input_output dataframe 
            
            inputs:
                combo_id:int
                data:strategydata object
                model: predictor object
                backtest backtest object
            '''
            
            self.input_output_df.loc[combo_id,'trainable_parameters'] = model.model.count_params()
            self.input_output_df.loc[combo_id,'training_records'] = len(data.features_train.index)
            self.input_output_df.loc[combo_id,'precision'] = np.mean(model.results.history['precision'][-5:])
            self.input_output_df.loc[combo_id,'accuracy'] = np.mean(model.results.history['accuracy'][-5:])
            self.input_output_df.loc[combo_id,'recall'] = np.mean(model.results.history['recall'][-5:])
            self.input_output_df.loc[combo_id,'gini'] = model.train_gini
            self.input_output_df.loc[combo_id,'test_gini'] = model.test_gini
            self.input_output_df.loc[combo_id,'test_accuracy'] = model.test_accuracy
            
            if self.input_output_df.loc[combo_id]['ml_type'] == 'DNN':
                self.input_output_df.loc[combo_id,'val_precision'] = np.mean(model.results.history['val_precision'][-5:])
                self.input_output_df.loc[combo_id,'val_accuracy'] = np.mean(model.results.history['val_accuracy'][-5:])
                self.input_output_df.loc[combo_id,'val_recall'] = np.mean(model.results.history['val_recall'][-5:])
                
            if self.input_output_df.loc[combo_id]['long'] == True:
                if data.labels_train['long_label'].value_counts().size == 2:
                    self.input_output_df.loc[combo_id,'train_label_freq'] = (data.labels_train['long_label'].value_counts()[1] 
                                                                    / data.labels_train['long_label'].size)
                if data.labels_test['long_label'].value_counts().size == 2:
                    self.input_output_df.loc[combo_id,'test_label_freq'] = (data.labels_test['long_label'].value_counts()[1] 
                                                                        / data.labels_test['long_label'].size)
                self.input_output_df.loc[combo_id,'gross_IR'] = backtest.gross_perf_metrics.loc['long']['IR']
                self.input_output_df.loc[combo_id,'gross_IR_baseline'] = backtest.gross_perf_metrics.loc['baseline_long']['IR']
                self.input_output_df.loc[combo_id,'net_IR'] = backtest.net_perf_metrics.loc['long']['IR']
                self.input_output_df.loc[combo_id,'net_IR_baseline'] = backtest.net_perf_metrics.loc['baseline_long']['IR']
            else:
                if data.labels_train['short_label'].value_counts().size == 2:
                    self.input_output_df.loc[combo_id,'train_label_freq'] = (data.labels_train['short_label'].value_counts()[1] 
                                                                    / data.labels_train['short_label'].size)
                if data.labels_test['short_label'].value_counts().size == 2:
                    self.input_output_df.loc[combo_id,'test_label_freq'] = (data.labels_test['short_label'].value_counts()[1] 
                                                                        / data.labels_test['short_label'].size)
                
                self.input_output_df.loc[combo_id,'gross_IR'] = backtest.gross_perf_metrics.loc['short']['IR']
                self.input_output_df.loc[combo_id,'net_IR'] = backtest.net_perf_metrics.loc['short']['IR']
        
        def run(self,combo_id,es,vx,show_backtest_charts=False,model_fit_stats=False, save_output=True, 
                defined_features = None):
            '''
            calibrate ML model over one combination of inputs and store results  
            
            Inputs:
                combo_id:int
                es: instrument class
                    es data
                vx: instrument class
                    vix data
                show_backtest_charts: boolean
                    show strategy backtest metrics and equity charts
                model_fit_stats: boolean
                    show model fit performance metrics (gini, roc, confusion matrix etc.)
                save_output: boolean
                    save performance to input_output dataframe
                defined_features:list
                    use user defined features 
                
            
            '''
            # prepare data
            _freq = self.input_output_df.loc[combo_id]['frequency']
            _include_vix = vx if self.input_output_df.loc[combo_id]['vix']== True else None
            
            data = StrategyData(inst = es,freq = _freq,vix = _include_vix)
            
            data.create_features()
            
            _clip = self.input_output_df.loc[combo_id]['clip']
            _date1 = self.input_output_df.loc[combo_id]['date1']
            _date2 = self.input_output_df.loc[combo_id]['date2']
            _date3 = self.input_output_df.loc[combo_id]['date3']

            _target = self.input_output_df.loc[combo_id]['target_label']
            _n = self.input_output_df.loc[combo_id]['n_label']
            
            # calculate standard deviation of 
            _t = data.inst.ticker
            _training_d = data.cleansed_data.loc[(data.cleansed_data.index > _date1) & (data.cleansed_data.index <= _date2)]
            _training_avg_c = _training_d['Close_'+_t].mean()
            _training_std = (np.log(_training_d['Close_'+_t] / _training_d['Close_'+_t].shift(1))).std()
            _target_std = _training_avg_c *  _training_std * _target * math.sqrt(_n)
            
            data.create_label(target_down=-1.*_target_std, target_up=_target_std,n=_n)
            self.input_output_df.loc[combo_id,'target_up'] =  _target_std
            self.input_output_df.loc[combo_id,'target_down'] = -1. * _target_std
            
            data.split_data_standardise([_date1,_date2,_date3],_clip)
            
            _start_trading_time = self.input_output_df.loc[combo_id]['start_trading_time']
            _end_trading_time = self.input_output_df.loc[combo_id]['end_trading_time']
            
            data.set_trading_hours(start_time=_start_trading_time,end_time=_end_trading_time)
            
            #ML model
            _ml_type = self.input_output_df.loc[combo_id]['ml_type'] 
            _epochs = self.input_output_df.loc[combo_id]['epochs']
            _long = self.input_output_df.loc[combo_id]['long']
            _hl = self.input_output_df.loc[combo_id]['hl']
            _hu = self.input_output_df.loc[combo_id]['hu']
            _dropout = self.input_output_df.loc[combo_id]['dropout']
            _rate = self.input_output_df.loc[combo_id]['rate']
            _run_boruta = self.input_output_df.loc[combo_id]['run_boruta']
            _drop_correlated_features = self.input_output_df.loc[combo_id]['drop_correlated_features']
            _optimiser = self.input_output_df.loc[combo_id]['optimiser']
            _lr = self.input_output_df.loc[combo_id]['lr']
            
            if _ml_type == 'DNN':
                _regularize = self.input_output_df.loc[combo_id]['regularize']
                _reg_l1 = self.input_output_df.loc[combo_id]['reg_l1']

                model = DNN(data, epochs=_epochs,hl=_hl,hu=_hu,
                            dropout=_dropout,rate=_rate,long = _long,
                            regularize=_regularize, reg_l1=_reg_l1, 
                            selected_features = defined_features)
                
                if defined_features is None:
                    if _run_boruta:
                        model.boruta_fit(data)
                        model.boruta_features_selection(data)
                    if _drop_correlated_features:
                        model.drop_correlated_features(data)
                
                if _optimiser == 'sgd':
                    _momentum = self.input_output_df.loc[combo_id]['momentum']
                    _nesterov = self.input_output_df.loc[combo_id]['nesterov']
                    model.set_optimiser(name = 'sgd', lr=_lr, 
                                        momentum=_momentum, nesterov=_nesterov)
                    model.fit(data)
                    model.predict(data,show_charts=model_fit_stats)
                else:
                    model.set_optimiser(name = _optimiser, lr=_lr)
                    model.fit(data)
                    model.predict(data,show_charts=model_fit_stats)
            
            elif _ml_type == 'DRNN':
                    _lags = self.input_output_df.loc[combo_id]['lags']
                    _layer = self.input_output_df.loc[combo_id]['layer']
                
                    model = DRNN(data,lags=_lags,hl=_hl,hu=_hu,layer=_layer,
                                dropout=_dropout,rate=_rate,epochs=_epochs,long=_long,
                                selected_features=defined_features)
                    
                    if defined_features is None:
                        if _run_boruta:
                            model.boruta_fit(data)
                            model.boruta_features_selection(data)
                        if _drop_correlated_features:
                            model.drop_correlated_features(data)
                    
                    if _optimiser == 'sgd':
                        _momentum = self.input_output_df.loc[combo_id]['momentum']
                        _nesterov = self.input_output_df.loc[combo_id]['nesterov']
                        model.set_optimiser(name = 'sgd', lr=_lr, 
                                        momentum=_momentum, nesterov=_nesterov)
                        model.fit(data,batch_size = 5,steps_per_epoch=10)
                        model.predict(data,show_charts=model_fit_stats)
                    else:
                        model.set_optimiser(name = _optimiser, lr=_lr)
                        model.fit(data,batch_size = 5,steps_per_epoch=10)
                        model.predict(data,show_charts=model_fit_stats)

                
            backtest = Strategy(data,model,model,show_chart=show_backtest_charts)
            
            if save_output:
                self.save_output(combo_id, data, model, backtest)
            
            return data, model, backtest
            
        def run_long_short(self,combo_id_long,combo_id_short,es,vx,show_backtest_charts=True,
                        defined_features_long = None, defined_features_short = None,
                        model_fit_stats=True):
            '''
            run long and short ML model and then perform back-test
            
            Inputs:
                combo_id_long:int
                    combo id of long approach
                combo_id_short:int
                    combo id of short approach
                es: instrument class
                    es data
                vx: instrument class
                    vix data
                show_backtest_charts: boolean
                    show performance statistics of back-test
                defined_features_long:list
                    use user defined features for long model
                defined_features_short:list
                    use user defined features for short model
            '''

            if self.input_output_df.loc[combo_id_long]['date1'] != self.input_output_df.loc[combo_id_short]['date1']:
                    print ('error : long and short sample not matching')
                    return
                
            if self.input_output_df.loc[combo_id_long]['date2'] != self.input_output_df.loc[combo_id_short]['date2']:
                    print ('error : long and short sample not matching')
                    return
            
            if self.input_output_df.loc[combo_id_long]['date3'] != self.input_output_df.loc[combo_id_short]['date3']:
                    print ('error : long and short sample not matching')
                    return

            
            
            data,model_long,_  = self.run(combo_id_long,es,vx,model_fit_stats=model_fit_stats,save_output=False, 
                                        defined_features = defined_features_long)
            
            _,model_short,_ = self.run(combo_id_short,es,vx,model_fit_stats=model_fit_stats,save_output=False,
                                        defined_features = defined_features_short)
            
            backtest = Strategy(data,model_long,model_short,show_chart=show_backtest_charts)
            
            return data, model_long, model_short, backtest

        
        def run_multiple(self,name,es,vx,defined_features=None, start_from_combo_id=None):
            '''
            run all combinations in the input_output dataframe and store results
            
            Inputs:
                name:str
                    name of combinations
                es: instrument object
                    es data
                vx: instrument object
                    vix data
                defined_features:list
                    use user defined features
                start_from_combo_id: int
                    if not None, the loop will start from comboid
            '''
            for i in self.input_output_df.index:
                if i<start_from_combo_id if start_from_combo_id is not None else 0:
                    continue
                self.run(i,es,vx, save_output=True, defined_features=defined_features)
                self.to_csv(name)
                self.input_output_df.to_pickle(name)
                print(str(i))
            
        
        def run_rolling(self,combo_id_long,combo_id_short,es,vx,months_backward,start_date=None):
            '''
            runs rolling ML models (one long and one short) that are recalibrated each month and 
            shows equity line
            
            Inputs:
                combo_id_long:int
                    combo id of long approach
                combo_id_short:int
                    combo id of short approach
                es: instrument class
                    es data
                vx: instrument class
                    vix data
                months_backward: int
                    number of months for rolling window for calibration
                start_date: datetime
                    user defined start date for rolling calibration
            '''
            #create months array for cross validation
            
            data = es.data if start_date == None else es.data[es.data.index>=start_date]
            
            start_date = data[:1].index.date[0]
            end_date = data[-1:].index.date[0]
            n_months = (end_date.year - start_date.year)*12 + (end_date.month - start_date.month) + 1 
            months = pn.date_range(start=start_date, periods = n_months, freq='1M')
            
            #dataframe to store strategy returns and performance stats
            strategy_gross_log_returns = pn.DataFrame()
            strategy_log_returns = pn.DataFrame()
            results = pn.DataFrame()
            long_pred_probabilities = pn.DataFrame()
            short_pred_probabilities = pn.DataFrame()
            
            #rolling cross validation
            for m in range(months_backward+1,len(months)):
                print(months[m])

                #update long model dates
                self.input_output_df.loc[combo_id_long,'date1'] = months[m - 1 - months_backward] 
                self.input_output_df.loc[combo_id_long,'date2'] = months[m - 1] 
                self.input_output_df.loc[combo_id_long,'date3'] = months[m]
                
                #update short model dates
                self.input_output_df.loc[combo_id_short,'date1'] = months[m - 1 - months_backward] 
                self.input_output_df.loc[combo_id_short,'date2'] = months[m - 1] 
                self.input_output_df.loc[combo_id_short,'date3'] = months[m]
                
                _,model_long, model_short, backtest = self.run_long_short(combo_id_long,combo_id_short,
                                                                        es,vx,show_backtest_charts=False,
                                                                        model_fit_stats=False)
                
                long_pred_probabilities = pn.concat([long_pred_probabilities, model_long.test_predict])
                short_pred_probabilities = pn.concat([short_pred_probabilities, model_short.test_predict])
                
                strategy_gross_log_returns = pn.concat([strategy_gross_log_returns, backtest.gross_log_returns])
                strategy_log_returns = pn.concat([strategy_log_returns, backtest.Net_log_returns])
                
                results.loc[months[m],'test_gini_long'] = model_long.test_gini
                results.loc[months[m],'test_gini_short'] = model_short.test_gini
                results.loc[months[m],'train_gini_long'] = model_long.train_gini
                results.loc[months[m],'train_gini_short'] = model_short.train_gini
                
                results.loc[months[m],'test_accuracy_long'] = model_long.test_accuracy
                results.loc[months[m],'test_accuracy_short'] = model_short.test_accuracy
                results.loc[months[m],'train_accuracy_long'] = model_long.train_accuracy
                results.loc[months[m],'train_accuracy_short'] = model_short.train_accuracy
                
                results.loc[months[m],'test_precision_long'] = model_long.test_precision
                results.loc[months[m],'test_precision_short'] = model_short.test_precision
                results.loc[months[m],'train_precision_long'] = model_long.train_precision
                results.loc[months[m],'train_precision_short'] = model_short.train_precision
                
                results.loc[months[m],'test_precision_recall'] = model_long.test_recall
                results.loc[months[m],'test_precision_recall'] = model_short.test_recall
                results.loc[months[m],'train_precision_recall'] = model_long.train_recall
                results.loc[months[m],'train_precision_recall'] = model_short.train_recall
            
            strategy_gross_log_returns.dropna(inplace=True)
            strategy_log_returns.dropna(inplace=True)
            
            perf_stats_net = backtest.calculate_perf_metrics(strategy_log_returns,backtest.max_bars_per_day)
            
            strategy_log_returns.cumsum().apply(np.exp).plot(figsize=(16, 6),title='net strategy returns');
            print(perf_stats_net) 
            
            self.rolling_perf_stats_net = perf_stats_net
            self.rolling_strategy_gross_log_returns = strategy_gross_log_returns
            self.rolling_strategy_log_returns = strategy_log_returns
            self.rolling_long_pred_probabilities = long_pred_probabilities
            self.rolling_short_pred_probabilities = short_pred_probabilities
            
            return results