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
- 1. Initial data preprocessing:
    - At first the raw data in the form of json is converted to dataframe .
    - Next, expanded urls and tweet creation timestamp are extracted and stored in another dataframe.
    - An additional excel sheet is provided which contains  media channels names with their sub urls and respective leanings.
    - Now from these dataset we need to regroup all left leaning media suburls into left, all right media suburls into right and all centre leaning media sub
    urls to centre. For that **HashMap** has been implemented in which keys are leaning and values are list of sub urls that are associated with their respective
    leaning.<br />
    - After iterating over whole additional excel sheet now hashmap has just 3 keys of left,right and centre with list of values of sub urls associated with them.
    - With this hashmap of 3 keys,3 different dataframes are generated in which we get left leaning media sub urls to one dataframe, right leaning media sub urls 
    to other dataframe and last dataframe contains centre leaning media sub urls.<br />
    -Later, main task here is to split timestamps from whole initial dataset to left,right and centre. For that to happen we use presence of sub urls
    in main urls and categorize them according to 3 dataframes that was created before.<br />
    - We will convert objects type of sub urls and expanded urls to string types in all dataframes.
    -As the intial dataframe is very huge it requires huge amount of time to iterate. So to make it faster we will first convert to vectorized array.
    -Now, by using re.find() we will check if respective sub url is present or not in main expanded urls. if present then we add time stamps to a list.
    -The above step is repeated for 3 times to get timestamps in 3 different lists of left,right and center.
    - At present we will have 3 lists of timestamps that are left,right and centre.(These lists have duplicates and that is required).
    -Getting how many duplicates for each timestamp in each list gives you frequency of respective leaning tweets per day. So to get frequency of each leaning tweets
    per day , we use *hashmaps* where in **iteration itself if one timestamp is not in key of hashmap then we put in it with value 1. if existed then 
    we increment respective timestamp value by 1**.
    - At this point of time we will be having 3 hashmaps with timestamps as keys and frequencies as values. These 3 hashmaps are left,right and center 
    timestamps wioth frequencies.
    -From this hashmap we will be converting to 3 dataframes .
 
 
- 2. Next step of data preprocessing:

