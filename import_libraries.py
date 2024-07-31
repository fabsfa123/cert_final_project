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
import winsound