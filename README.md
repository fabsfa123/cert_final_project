# CPF Final Project - Fabio Del Bo

The repository contains all the required python and data files to run the CPF Final Project on **“Deep Learning methods to forecast price movements in S&P 500 Futures”** by Fabio Del Bo. The project is structured and documented such that a Jupiter Notebook is executable on Google Colab.


# Abstract 

Deep Learning techniques are assessed as signal generators for systematic trading strategies on the S&P 500 Futures. Firstly, Dense Neural Network and Recurrent Neural Networks are applied to classify future movements in intraday prices using indicators derived from historical intraday prices and volumes of the S&P 500 E-Mini Futures (ES) and the Cboe Volatility Index (VIX). Forecasts of positive and negative price changes based upon two distinct models are going to be evaluated with the ambition of factoring the asymmetry in equity returns. The predictive performance of the classification models is assessed via cross validation over different hyperparameters combinations in order to achieve a satisfactory bias-variance trade-off. The forecasts are then used to back-test a trading strategy on the S&P 500. To conclude, the application of the explainability package (LIME) is going to be considered as an interesting solution to explain the key drivers of advanced machine learning forecasts. 

# How to set up on Google Colab



 - 1. Open the Jupyter Notebook on google colab:
    ```bash
    


 - 2. Open the Jupyter Notebook *my_project.ipynb* on folder */content/cert_final_project* 
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









conda activate project2   
jupyter notebook   
open MyProject   