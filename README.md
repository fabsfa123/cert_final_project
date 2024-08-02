# CPF Final Project - Fabio Del Bo

The repository contains all the required python and data files to run the CPF Final Project on **“Deep Learning methods to forecast price movements in S&P 500 Futures”** by Fabio Del Bo. The project is structured and documented such that a Jupiter Notebook is executable on Google Colab.


# Abstract 

Deep Learning techniques are assessed as signal generators for systematic trading strategies on the S&P 500 Futures. Firstly, Dense Neural Network and Recurrent Neural Networks are applied to classify future movements in intraday prices using indicators derived from historical intraday prices and volumes of the S&P 500 E-Mini Futures (ES) and the Cboe Volatility Index (VIX). Forecasts of positive and negative price changes based upon two distinct models are going to be evaluated with the ambition of factoring the asymmetry in equity returns. The predictive performance of the classification models is assessed via cross validation over different hyperparameters combinations in order to achieve a satisfactory bias-variance trade-off. The forecasts are then used to back-test a trading strategy on the S&P 500. To conclude, the application of the explainability package (LIME) is going to be considered as an interesting solution to explain the key drivers of advanced machine learning forecasts. 

# How to set up on Google Colab



 - 1. Open the Jupyter Notebook on Google Colab by clicking on the link below:
    ```bash
    https://github.com/fabsfa123/cert_final_project/blob/5a1998499abf55db23f20a1b6ab37f1436cd2076/my_project.ipynb
 - 2. Clone the repository 
    ```bash
    git clone https://fabsfa123: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@github.com/fabsfa123/cert_final_project.git
 - 3. Install condacolab and ta-lib
    ```bash
    !pip install -q condacolab
    import condacolab
    condacolab.install()
    !conda install -c conda-forge ta-lib lime
- 4. pip install tensorflow (ol;d version), boruta, plotly, and lime 
    ```bash
    !pip install tensorflow==2.12 boruta plotly lime
- 5. you will be promtled to restart the kernel. proceed
- 6. run the jupyter notebook


# Documentation and Structure

-	**my_project.ipynb** is the Jupyter Notebook that executes and present the essay. This is in essence the main file of the project
-	**my_classes** is a .py file containing all the classes that I created to support the analysis of the project.  The classes are documented and can be listed as:
-	**Instrument**:  instrument's historical data and characteristics and charting methods
-	**StrategyData**: methods and data to cleans data for model calibration and back-testing. Inherits from Instrument. 
    -   *Predictor*: neural networks base class
    -	*DNN*: methods and data related to DNN. It inherits from Predictor
    -	*DRNN*: methods and data related to DRNN. It inherits from Predictor
    -	*Strategy*: vectorised back-testing
    -	*cross_validation*: various cross validation
-	**Input Data**
    -	*ES_5min_continuous_adjusted.txt*: historical S&P500 futures data 
    -	*VX_full_5min_continuous_Unadjusted*: historical VIX futures data
-	**Stored Outputs**: if the user wants to analyse the output without rerunning all the cross section methods, the files below can be used by commenting the relevant lines. Otherwise, the model can be ran entirely, but it will take longer
    - *dnn_adam*: pickle file with cross_section object with all the results of the in sample cross sectional analysis using dnn_adam 
    - *dnn_adam.csv*: results of dnn_adam cross sectional analysis
    - *dnn_sgd*: pickle file with cross_section object with all the results of the in sample cross sectional analysis using dnn_sgd
    - *dnn_sgd.csv*: results of dnn_sgd cross sectional analysis
    - *drnnv2*: pickle file with cross_section object with all the results of the in sample cross sectional analysis using dnn_sgd
    - *drnnv2.csv*: results of drnn cross sectional analysis
    - *log_rets_drnn.csv*: returns of the out of sample forward testing trading strategy based on drnn prediction
    - *rolling_dnn_adam_rets.csv*: returns of the out of sample forward testing trading strategy based on dnn adam prediction
    - *rolling_dnn_sgd_rets.csv*: returns of the out of sample forward testing trading strategy based on dnn sgd prediction
