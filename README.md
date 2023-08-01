# Forecasting of Political Bias in Twitter and GAB.

## Introduction.

In order to grasp the attention of the huge public, amalgamating polarized content in Twitter and GAB
by respective news channels has become a quotidian.<br />
With this propensity of news media, there have been thousands of tweets, and GAB posts circulating
every day which are leaned in either of the 5 types: left, right, center, left-leaning, and right-leaning depending 
upon the leaning of their respective media channel.<br />
The fact that there are enormous numbers of tweets and GAB posts generating per day raises a very much 
challenging task of knowing how much political bias content can be 
generated in the next day or the next few days.<br />
***This project aims to deal with this challenge by implementing 5 of the time-series 
forecasting methods of Machine learning which predicts the behavior of any quantity in
next day or next few days by taking the previous days' behavior.***<br />

## Gist of used Timeseries forecasting models.

|**Model**                   | **Definition**                 |*No.of lookbacks(Previous days count)* | *No.of next days predicted*|
|----------------------------|--------------------------------|---------------------------------------|----------------------------|
| SARIMA                     |A statistical method used       |                  0                    |              1             |
|                            |to forecast time series         |                                       |                            |
|                            |based on the average of lags and|                                       |                            |
|                            | seasonality trends of          |                                       |                            |
|                            |previous time steps of data.    |                                       |                            |
| LSTM-I                     |A deep learning method which is |                  1                    |              1             |
|                            |an extension of RNN that is     |                                       |                            |
|                            |designed to learn a sequential  |                                       |                            |
|                            |data by preserving them.        |                                       |                            |
| LSTM-II                    |Same as LSTM-I but only         |                 14                    |              1             |
|                            |difference is in LSTM-I we use  |                                       |                            |
|                            | Only one previous day for      |                                       |                            |
|                            |predicting next day but here    |                                       |                            |
|                            | We can use as many lookbacks   |                                       |                            |
|                            |as we want for the next day.    |                                       |                            |
| MULTISTEP FORECASTING      |A LSTM model,combination of     |                 14                    |              7             |
|                            |Above two models in which we can|                                       |                            |
|                            |use as many previous days'      |                                       |                            |
|                            |patterns to predict next        |                                       |                            |
|                            |sequence of days.               |                                       |                            |
|                            |                                |                                       |                            |
| GRU                        |A simplified version of LSTM in |                 14                    |              1             |
|                            |which it has only 2 gates       |                                       |                            |
|                            |reset and update whereas        |                                       |                            |
|                            |LSTM has 3 gates(input, output  |                                       |                            |
|                            |and forget gates.               |                                       |                            |


**NOTE: Multiple lookbacks are done vector concatenation and given as input for the models which have 14 days of lookback. And for predicting next 7 days in Multistep time-series forecasting a method called teacher enforcing (Teacher forcing is a method for quickly and efficiently training recurrent neural network models that use the ground truth from a prior time step as input) is established internally which uses previous day's ground truth as input for today and this process continues for vector of 14 days.**

## Methodology :
- ### 1. Initial data preprocessing:
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
    we increment respective timestamp value by 1**.<br />
    - At this point of time we will be having 3 hashmaps with timestamps as keys and frequencies as values. These 3 hashmaps are left,right and center 
    timestamps wioth frequencies.<br />
    - From this hashmap we will be converting to 3 dataframes .<br />
 
 

- ### 2. Next step of data preprocessing:
    - As we have left,right and centre timestamps with frequencies we take range of required data, here timestamps between may to december 2018 have taken
    and converted to timeseries.
    

- ### 3. Application of grid search for SARIMA parameters and SARIMA MODEL FITTING.
    - Grid search is a algorithm with wich we can select parameters for the model. It automatically tries all the combinations of SARIMA parameters for the dataset
    and gives the best combination which showed less RMSE. With these parameters we fit 3 timeserieses to SARIMA model and we will be getting
    3 different RMSEs for left, right and centre leaning.
    
- ### 4. Keras tunning and application of LSTM-1, LSTM-2 and Multistep forecasting.
    - The timeserieses have been splited to 67,33 or 80,20 percentages of training and testing datasets and applied keras tunner which gives best combination
    of LSTM parameters with low rmse of all.
    - After every time running keras tunner for left timeseries, right timeseries and centre time series with respective derived hypermaters LSTM-1, LSTM-2 and
    multistep forecasting are trained and predicted.
    
- ### 5. Conclusion.
    - All these models are compared based on **RMSE'S (Root Mean Squared Errors)** of all left, right and center timeseries after fitting to above models.

## Note:

- ### 1. Same methodolgy has been applied to 2 datasets of json format which are 4.5 gb and 2.5 gb of sizes.<br />
- ### 2. 2 tasks have been implemented with this methodology and second task is more specified version of task-1. Task-1 is predicting polarized tweet next or few           days, whereas task-2 is adaption rate predicting likes count , shares count or dislikes count.<br />
- ### 3. The only difference between task-1 and task-2 is that for task-2 in addition with expanded urls, we extract likes count,shares count and dislikes count.


  

