# Forecasting of Political Bias in Twitter and GAB.

## Introduction.
In this study, we embark on an innovative research venture aimed at predicting political bias in online social media platforms, such as Twitter and Gab. We frame this as a time series forecasting challenge, where we input a time series of politically biased data and the forecasting model predicts future developments or the next steps in the series. By accurately anticipating how political sentiments evolve over time, we seek to gain insights into the formation of ideological groups and the dissemination of biased information within social media platforms. This knowledge can be invaluable in devising effective strategies to combat misinformation and echo chamber effects. It's worth noting that previous research in forecasting political bias data is limited, making our exploration groundbreaking in this field. We utilize existing time series forecasting models to assess their suitability for this task. Through an analysis of how well these models forecast the time series of political leanings, we aim to uncover their strengths and weaknesses in capturing the temporal dynamics of political bias. As each social media platform has its unique user engagement, popularity, and political ideologies, we conduct experiments using two social media datasets, Twitter and Gab, collected during the same timeframe. In summary, our paper offers two main contributions:

***Contribution-1: We introduce a novel challenge of predicting political bias in online social media posts, which is pivotal for understanding the political stance of social media on various topics or events.*** <br />
***Contribution-2: We experiment with diverse time series forecasting models to quantify trends in different political biases within a social media platform. In our analysis, we examine two social media platforms with distinct political biases: Twitter and Gab***

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


## Datasets :
In this study, we made use of publicly accessible datasets obtained from Twitter in 2018 and Gab in 2018. The Twitter dataset comprises tweets that contain links to news articles related to political topics from selected news sources. This dataset covers the period from January 2018 to October 2018 and includes a total of 722,685 tweets. On the other hand, our Gab dataset from 2018 is extensive, containing 40 million posts, including replies, reposts, and quotes, all featuring URLs and hashtags. The data spans from 2016 to 2018. To ensure a fair analysis, we only considered a subset of Gab posts that shared news article URLs during the period from January 2018 to October 2018, which corresponds to the time frame of the Twitter data. It's worth noting that unlike Twitter, Gab data isn't specifically concentrated on any particular subject, which accounts for its larger number of posts. <br/>

## Implementation of Methodology:
- ### 1. Initial data preprocessing:
    - At first, the raw data in the form of JSON is converted to a data frame.
    - Next, expanded URLs and tweet creation timestamps are extracted and stored in another data frame.
    - An additional Excel sheet is provided which contains media channel names with their sub urls and respective leanings.
    - Now from these datasets, we need to regroup all left-leaning media suburbs into left- left-leaning, all right media suburbs into right, all centre-leaning media sub
    URLs to center, all left media to left, and all right-leaning media to right-leaning. For that,  **HashMap** has been implemented in which keys are leaning and values are the list of sub URLs that are associated with their respective
    leaning.<br />
    - After iterating over the whole additional Excel sheet now hashmap has just 5 keys of left, right, center, left-leaning, and right-leaning with a list of values of sub URLs associated with them.
    - With this hashmap of 5 keys,5 different data frames are generated in which we get left-leaning media sub URLs to one data frame, right-leaning media sub urls 
      to another data frame, right media sub urls to another data frame, left media sub urls to another data frame, and center media URLs to another data frame .<br />
    -Later, the main task here is to split timestamps from the whole initial dataset to left, right, center,left-leaning, and right-leaning. For that to happen we use the presence of sub urls
    in main urls and categorize them according to 5 dataframes that were created before.<br />
    - We will convert the objects type of sub urls and expanded URLs to string types in all data frames.
    -As the initial data frame is very huge it requires a  huge amount of time to iterate. So to make it faster we will first convert to vectorized array.
    -Now, by using re .find() we will check if the respective sub URL is present or not in the main expanded urls. if present then we add timestamps to a list.
    -The above step is repeated for 5 times to get timestamps in 5 different lists of left, right, center, left-leaning, and right-leaning.
    - At present, we will have 5 lists of timestamps that are left, right, center, left-leaning, and right-leaning. (These lists have duplicates and that is required).
    -Getting how many duplicates for each timestamp in each list gives you the frequency of respective leaning tweets per day. So to get the frequency of each leaning tweets
    per day, we use *hashmaps* where in **iteration itself if one timestamp is not in the key of hashmap then we put in it with value 1. If existed then 
    we increment the respective timestamp value by 1**.<br />
    - At this point in time we will be having 5 hashmaps with timestamps as keys and frequencies as values. These 5 hashmaps are left, right, center, left-leaning, and right-leaning 
    timestamps with frequencies.<br />
    - From these hashmaps, we will be converting to 5 data frames .<br />
 
 

- ### 2. Next step of data preprocessing:
    - As we have left, right, center, left-leaning, and right-leaning timestamps with frequencies we take a range of required data, here timestamps between May to October 2018 have taken
    and converted to time series because of outliers and fewer postings from January to May. <br />
    

- ### 3. Application of grid search for SARIMA parameters and SARIMA MODEL FITTING.
    - Grid search is an algorithm with which we can select parameters for the model. It automatically tries all the combinations of SARIMA parameters for the dataset
    and gives the best combination which showed less RMSE. With these parameters, we fit 5 time series to the  SARIMA model and we will be getting
    5 different RMSEs for left, right, center, left-leaning, and right-leaning.
    
- ### 4. Keras tunning and application of LSTM-1, LSTM-2, GRU, and Multistep forecasting.
    - The time series have been split in a window of 60-80, 20-40 percentages of training and testing datasets, and applied Keras tuner which gives the best combination
    of LSTM parameters with low rmse of all.

## Outputs :-
The outputs folder contains 4 images of multi-line plots conveying which average tweet frequency or Gab posts, and total Tweet likes or Gab post likes per month, 2 images of stacked plots in which 1st image contains the Average Compound score per day of tweet content, and another image contains the Average Compound score per day of Gab post content. These are the results that were obtained after applying VADER sentiment analysis. And 2 Excel sheets in which one is  RMSES of Twitter posts, likes and another one is RMSES of Gab posts and their likes which were recorded after forecasting with above models.
    
    
- ### 5. Conclusion.
    - All these models are compared based on **RMSE'S (Root Mean Squared Errors)** of all left, right, center, left-leaning, and right-leaning time series after fitting to the above models.

## Note:

- ### 1. The same methodology Implementation has been applied to the GAB dataset as well but the way of preprocessing will be different because of the different JSON format.<br />
- ### 2. 2 tasks have been implemented with this methodology and the second task is a more specified version of task-1. Task-1 is predicting polarized tweets or Gab posts next few days, whereas task-2 is adaption rate which is predicting Tweet likes or Gab post likes.<br />
- ### 3. The only difference between task-1 and task-2 is that for task-2 in addition to expanded urls, we extract likes count.


  

