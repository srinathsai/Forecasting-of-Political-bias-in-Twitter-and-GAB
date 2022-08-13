# Prediction-of-Polarized-Tweets

## Introduction.

In order to grasp the attention of huge public, amalgamating polarized content in tweets
by respective news channel has become a quotidian.<br />
With this propensity of news media, there has been thousands of tweets circulating
every day which are leaned in either of the 3 types: left,right and center depending 
upon the leaning of their respective media channel.<br />
The fact that there are enormous number of tweets generating per day raises a very much 
challenging task of knowing how much polarized tweets can be 
generated in next day or next few days.<br />
***This project aims to deal with this challenge by implementing 4 of the timeseries 
forecasting methods of Machine learning which predicts the behavioir of any quantity in
next day or next few days by taking previous days behaviour.***<br />

## Gist of used Timeseries forecasting models.

|**Model**                   | **Definition**                 |*No.of lookbacks(Previous days count)* | *No.of next days predicted*|
|----------------------------|--------------------------------|---------------------------------------|----------------------------|
| SARIMA                     |A statistical method used       |                  0                    |              1             |
|                            |to forecast time series         |                                       |                            |
|                            |based on  average of lags and   |                                       |                            |
|                            | seasonality trends of          |                                       |                            |
|                            |previous time steps of data .   |                                       |                            |
| LSTM-I                     |A deep learning method which is |                  1                    |              1             |
|                            |an extension of RNN that is     |                                       |                            |
|                            |designed to learn a sequential  |                                       |                            |
|                            |data by preserving them.        |                                       |                            |
| LSTM-II                    |Same as LSTM-I but only         |                 2,3                   |              1             |
|                            |difference is in LSTM-I we use  |                                       |                            |
|                            |only one previous day for       |                                       |                            |
|                            |predicting next day but here    |                                       |                            |
|                            |we can use as many lookbacks    |                                       |                            |
|                            |as we want for next day.        |                                       |                            |
| MULTISTEP FORECASTING      |A LSTM model,combination of     |                 2,3                   |               6            |
|                            |Above two models in which we can|                                       |                            |
|                            |use as many previous days patern|                                       |                            |
|                            | to predict as many next days.  |                                       |                            |

**NOTE: As this project has been implemented over 2 datasets of different sizes, after applying keras tunner for hyper parameterization and various trail
ans errors no of look backs found ideal for below 2 models to be 2 for dataset-1 and 3 for dataset-2.**

## Methodology :


