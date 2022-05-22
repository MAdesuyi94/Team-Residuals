# Humana - Mays Case Competition – 2021
# Classification of Individuals Hesitant to the COVID-19 Vaccine using Modern Machine Learning Methods

## Overview
  Humana is dedicated to reducing barriers of access to the COVID-19 vaccine for the socially vulnerable, as well as prioritizing effective outreach and engagement with the general population leading to increased overall vaccination rates within the community. 
  
  At the time of writing (October 2021), there have been approximately 44.3 million reported cases of COVID-19 in the United States, and 713,000 deaths.  COVID-19 vaccines are effective at protecting patients from COVID-19, especially severe illness and death. In addition, COVID-19 vaccines can reduce the risk of people spreading the virus that causes COVID-19.  A June 14, 2021 analysis by the Washington Post found that Coronavirus infection rates are lower in communities where people are vaccinated and higher where they are not.  It stands to reason then, that Humana and other health insurance providers would pay careful attention to the differences in characteristics between the vaccinated and unvaccinated populations. Understanding these differences and being able to predict which members are hesitant to receiving the COVID-19 vaccine will help Humana build targeted plans for outreach, communication, and education.
  
  The goal is to better understand the defining characteristics of each group for Humana to strategically engage those who have not received the vaccine. Do they require more information on the vaccines in the market? Do they have health concerns? Are they physically unable to receive the vaccine due to pre-existing conditions? It is unclear what drives people to receive the vaccine against COVID-19, however this paper will explore the data provided to come to a viable conclusion on how to engage those who remain unvaccinated. The business question we attempt to answer using machine learning methods is: what are the factors that influence COVID-19 vaccine hesitancy in Humana patients and in the general population? 
  
  The dataset used for the analysis was provided by Humana with various characteristic data they have collected on their customers. Humana is a healthcare insurance provider in the United States of America with at least 4.5 million members according to a recent article in June 2020 on Humana’s website. The data set provided contained at least 1 million customer’s personal information, implying that at least 22% of customers are represented in the data.

## Steps to Building a Viable Model
Creating a model with an appropriate level of sensitivity and specificity is key to understanding Humana’s customers. During the modeling process, the following steps were performed in order to propose a solution:
 

1.	A thorough understanding of the problem we are trying to solve, and it’s impacts to Humana
2.	Preliminary data exploration with visualization to fully understand the data and evaluate outliers
3.	Data cleansing and preprocessing to create consistent data types in addition to imputing missing data and encoding non-numeric data
4.	Fitting various machine learning models using the cleansed and processed data 
5.	Comparing and evaluating the fitted models to determine the most parsimonious model
6.	Using the best and final model to make well-informed recommendations for Humana

## Exploratory Analysis
The team used a combination the analytical tools below to import, explore, visualize, clean, process, and model the data. 

•	MS SQL Server

•	SAS Enterprise Miner

•	JMP

•	Python

## Data Exploration
In MS SQL Server, a simple query was used for each of the variables (example below) to gain a high level to the percentages for a specific value for that variable. This process was helpful in identifying features with a high percentage of NULL values and where the values were heavily skewed on a single value. Additionally, it quickly showed us that the dataset had a significantly higher proportion of non-vaccinated individuals (82.6%) in compared to vaccinated individuals (17.4%) in the target class. While not necessarily a rare event problem, we would have to factor this large proportion difference into the rest of our analysis.

![SQL Server Screenshot](https://github.com/MAdesuyi94/Team-Residuals/blob/MAdesuyi94-patch-1/Images/SQL%20Screenshot.PNG)

## Data Visualization
First, each variable was examined based on its distribution. Distribution plots were created for each numeric variable and assessed for normality, also noting any specific outliers. 

![Histograms](https://github.com/MAdesuyi94/Team-Residuals/blob/MAdesuyi94-patch-1/Images/Histograms.png)

Next, box plots were created for the numeric variables to search for any visual relationships against the target variable. Specifically, we were looking for features that showed differences in the distributions and average values when comparing the two target classes.
 
 ![Boxplots](https://github.com/MAdesuyi94/Team-Residuals/blob/MAdesuyi94-patch-1/Images/Boxplots.png)

Count plots were evaluated for categorical features, searching for categories that show significant differences between the target class.
 
![Countplots](https://github.com/MAdesuyi94/Team-Residuals/blob/MAdesuyi94-patch-1/Images/Countplots.png)

Density plots were also evaluated to examine the differences in distributions from vaccinated to unvaccinated individuals. 

![Densityplots](https://github.com/MAdesuyi94/Team-Residuals/blob/MAdesuyi94-patch-1/Images/Density%20plots.png)

## Data Cleansing and Preprocessing
Examining the shape of the training dataset provided by Humana, there were initially 366 predictive features, one binary target variable (covid_vaccination), and 974,842 observations; each observation representing a Humana MAPD member. 
The predictive features in the training dataset fell into one of the following categories:

•	Medical Claims

•	Pharmacy Claims

•	Lab Claims

•	Demographics/Consumer Data

•	Credit Data

•	Condition Related

•	CMS

•	Other

The data were also categorized based on the data types below. 

•	Numeric – Continuous: 

  o	Typically seen with credit data and demographics/consumer data
  
•	Numeric – Discrete

  o	Count data, usually reporting unique authorizations for admits of distinct types
  
•	Nominal

  o	Trend data and other unique categorical classifications
  
•	Binary 

  o	Indicator values where 1 = positive indicator and 0 = negative indicator
  
Many of the features (both numeric and categorical) had missing values as well as observations recorded as ‘*’. The ‘*’ values did not fit within the framework of the data types and were replaced with missing values for imputation. 

## Feature Selection
Using 25% as a cutoff point for missing values, the team elected to drop the following six features from the training dataset. Although many may have intuitively been useful for adding to the predictive value of these models, such as the MAPD behavioral segment or the preferred language for the member, there were simply too many missing values to ensure that imputation would not lead to a spurious relationship with the target variable. 

![Feature Selection Table](https://github.com/MAdesuyi94/Team-Residuals/blob/MAdesuyi94-patch-1/Images/Feature%20Selection%20Table.PNG)

Additionally, each remaining variable was carefully examined for its basic descriptive statistics, including the data type, mean, standard deviation, minimum, 25%, 50%, 75%, and maximum values. Using this evaluation, another 54 variables were dropped from the analysis considering their low variance threshold (variance = 0). These zero variance features have no predictive value to the target class and will not result in an increase to predictive power, only to the complexity of the model. 

## Imputation
For the remaining features with missing data, the team decided to impute the mean value for numeric features and the most frequent value for the categorical and binary features.

## Encoding
Categorical features were encoded as numeric data using one-hot encoding, which creates a binary dummy variable for each category within the column. Doing so increased the size of the dataset significantly from 313 columns to 779 columns.

## Important Observations
As mentioned previously, the original training data had an imbalance of vaccinated to unvaccinated individuals with 82.6% in the unvaccinated class and 17.3% in the vaccinated class. Many machine learning models, especially tree-based models, will bias their results to favor such an imbalance. Due to the imbalance of the target class, it was deemed strategic to split the training dataset into training and validation sets and use resampling techniques. The dataset was split using stratified sampling to account for the proportion of vaccinated and unvaccinated subjects, and the stratified sample generated training and validation sets with 50% unvaccinated versus 50% vaccinated persons.  For the training/validation split, the team used a combination of 60/40 and 70/30 to begin building and exploring models. 

![Vaccination Counts](https://github.com/MAdesuyi94/Team-Residuals/blob/MAdesuyi94-patch-1/Images/Vaccination%20counts.png)

![Balanced 70/30 Split](https://github.com/MAdesuyi94/Team-Residuals/blob/MAdesuyi94-patch-1/Images/Balanced%207030%20split.png)

![Balanced 60/40 Split](https://github.com/MAdesuyi94/Team-Residuals/blob/MAdesuyi94-patch-1/Images/Balanced%206040%20split.png)

## Model Proposal
## Testing Various Models
## Random Forest in JMP
Random forest is an ensemble machine learning model that utilizes a plethora of trees to construct a powerful predictive model. The stratified sample of the original training data provided was utilized in order to build a bootstrapped random forest model. The stratified sample had a more equal distribution of the people vaccinated and unvaccinated for COVID-19. The optimal model appeared to be a random forest with at least 56 trees. 

The AUC for the stratified training data set was 0.70 and 0.6675 for the validation set. However, since the goal is to better understand who is unvaccinated it becomes important to look at the model’s specificity, or true negative rate. In the training data it was 57% and in the validation it was about 55%. The model’s sensitivity, or how many times our model correctly predicted vaccinations, was conversely with 71% in training and 69% in validation. The model is good at predicting if a person is vaccinated rather than unvaccinated; However, the model’s misclassification rate was about 35.6% for the training data and 37.8% for the validation data. This indicates that random forest is able to predict, with some degree of sensitivity, vaccinations and does not often misclassify vaccinated and unvaccinated persons. 

![Bootstrap Forest Results](https://github.com/MAdesuyi94/Team-Residuals/blob/MAdesuyi94-patch-1/Images/Bootstrap%20forest%201.png)

![Bootstrap Forest ROC](https://github.com/MAdesuyi94/Team-Residuals/blob/MAdesuyi94-patch-1/Images/Bootstrap%20ROC.png)

## XGBoost
XGBoost, also known as Extreme Gradient Boosting, is an ensemble machine learning model that utilizes many decision trees to develop a final model. Unlike random forest, each subsequent tree is applied a small learning rate to reduce the error in the overall model prediction. Also, regularization terms, such as lambda and gamma, are incorporated to prevent each tree from unnecessarily growing to the full depth (which in turn prevents overfitting). One of the biggest advantages of XGBoost is that it can handle unbalanced data very well. The parameter scale_pos_weight gives the ability to add more weight to the minority samples (or decrease the weight of the majority samples), eliminating the need for Random Under Sampling. This is favorable because no data will need to be deleted to train the model. 

```python
param_grid = {
        "max_depth": [4,5,10,15,20,25,30],
        "learning_rate": [0.05,0.1,0.3,0.5],
        "gamma": [0,0.25,1],
        "reg_lambda": [0,1,10],
        "scale_pos_weight": [0.2],
        "subsample": [0.3,0.5,0.8],
        "colsample_bytree": [0.5,0.7,0.8]
        }
```

A grid search was utilized in order to find the best hyperparameters to use for the model. Along with the grid search, 3-fold cross validation was also used to ensure that the best tuning parameters gave higher results on average. The parameters utilized are displayed in the picture above. The results on the grid search displayed that the following parameters were deemed as optimal:

•	Max Depth: 4

•	Learning Rate: 0.3

•	Gamma: 1

•	Lambda: 1

•	Subsample: 0.3

•	Columns Sampled by Tree: 0.7

In addition to the parameters described above, a high tree count of 5000 was utilized in the model. To prevent overfitting due to a large amount of trees, an option was set to discontinue training if 15 trees were built without any impovement on the training set. After the optimal parameters were chosen, the model was ran on the entire training and validation sets. The results are shown below. 

![XGBoost Training Results](https://github.com/MAdesuyi94/Team-Residuals/blob/MAdesuyi94-patch-1/Images/XGBoost%20Training%20Results.png) ![XGBoost Test Results](https://github.com/MAdesuyi94/Team-Residuals/blob/MAdesuyi94-patch-1/Images/XGBoost%20Test%20Results.png)

![XGBoost Training Confusion Matrix](https://github.com/MAdesuyi94/Team-Residuals/blob/MAdesuyi94-patch-1/Images/XGBoost%20Training%20Confusion%20Matrix.png)     ![XGBoost Test Confusion Matrix](https://github.com/MAdesuyi94/Team-Residuals/blob/MAdesuyi94-patch-1/Images/XGBoost%20Test%20Confusion%20Matrix.png)

![XGBoost ROC Curve](https://github.com/MAdesuyi94/Team-Residuals/blob/MAdesuyi94-patch-1/Images/XGBoost%20ROC.png)

Based on the results displayed above, XGBoost does a good job of not overfitting the model. As seen on the ROC curve and the results above, the AUC for the test set (0.6748) is only slightly lower than the AUC for the training set (0.69). By looking at the confusion matrix, it does a decent job of identifying unvaccinated individuals. Although the misclassification rate (42%) is a bit higher than the other models ran, the AUC is higher than the random forest model. This is important, as it can help differentiate between vaccinated and unvaccinated individuals. 

## LightGBM
LightGBM, also known as Light Gradient Boosting Machine, is another ensemble machine learning model that utilizes numerous trees to develop a final model. Similar to XGBoost, LightGBM also incorporates a learning rate to subsequent trees to improve predictions and regularization to prevent trees from overfitting. Unlike XGBoost, the trees are grown in a leaf-wise fashion as opposed to a depth-wise fashion. This enables LightGBM to run much faster than XGBoost. Another advantage of LightGBM over XGBoost is its ability to handle categorical features. To be specific, LightGBM accepts categorical variables as long as they’re labeled as integers (known as label encoding). In comparison, XGBoost requires categorical features to be one-hot encoded, which can create sparse data (250% more columns) and potentially yield less-optimal results. 

```python
param_grid = {
    "num_leaves":       [31,50,70],
    "learning_rate":    [0.1,0.05,0.02,0.01],
    "class_weight":     [{1:1,0:5}],
    "bagging_fraction": [0.8,0.5,0.3],
    "colsample_bytree": [0.8,0.7,0.5,0.3]
    }
```

In order to find the best model that gave consistent optimal parameters, 3-fold cross-validation was utilized along with grid search of the best hyperparameters. The parameters used in the grid search are located in the picture above. The results on the grid search displayed that the following parameters were deemed as optimal:

•	Max Leaves: 31 (Default)

•	Learning Rate: 0.02

•	Bagging Fraction: 0.3

•	Columns Sampled by Tree: 0.1

In addition to the parameters selected by the grid search, the max number of trees estimated was set to 5000, and the number of rounds before stopping early due to lack of improvement was set to 15. This enables the model to learn with smaller learning rates and prevent the training process from stopping prematurely, and it also enables the model to stop running if there is lack of improvement. After the optimal parameters were chosen, the model ran on the entire training set and score on the validation set to yield the following results:

![LightGBM Training Results](https://github.com/MAdesuyi94/Team-Residuals/blob/MAdesuyi94-patch-1/Images/LightGBM%20Training%20Results.png)      ![LightGBM Test Results](https://github.com/MAdesuyi94/Team-Residuals/blob/MAdesuyi94-patch-1/Images/LightGBM%20Validation%20Results.png)

![LightGBM Training Confusion Matrix](https://github.com/MAdesuyi94/Team-Residuals/blob/MAdesuyi94-patch-1/Images/LightGBM%20Training%20Confusion%20Matrix.png)     ![LightGBM Test Confusion Matrix](https://github.com/MAdesuyi94/Team-Residuals/blob/MAdesuyi94-patch-1/Images/LightGBM%20Validation%20Confusion%20Matrix.png)

![LightGBM ROC Curve](https://github.com/MAdesuyi94/Team-Residuals/blob/MAdesuyi94-patch-1/Images/LightGBM%20ROC.png)

Based on the above metrics, the model appears to avoid overfitting. For example, the AUC for the training set is .7073, and the AUC for the test set is .6834. The model seems to do a decent job of identifying vaccinated and unvaccinated. Although the misclassification is a bit higher than the other models, the AUC is the highest out of all the models. It is easy to get an extremely low misclassification rate by having a model just predict all the subjects as unvaccinated; However, that model would be rendered useless (along with a low AUC score) because it fails to differentiate between the vaccinated and unvaccinated individuals. 

![LightGBM Feature Importance](https://github.com/MAdesuyi94/Team-Residuals/blob/MAdesuyi94-patch-1/Images/LightGBM%20Feature%20Importance.png)

Based on the feature importance graph, age appears to be one of the top important factors in determining individuals who are hesitant to take the vaccine. 

![Vaccination Boxplot 1](https://github.com/MAdesuyi94/Team-Residuals/blob/MAdesuyi94-patch-1/Images/Vaccination%20vs%20Age%20Boxplot.png)      ![Vaccination Boxplot 2](https://github.com/MAdesuyi94/Team-Residuals/blob/MAdesuyi94-patch-1/Images/Vaccination%20vs%20Payment%20Boxplot.png)

As seen in the boxplot and the importance graph, unvaccinated individuals seem to pay a higher amount on average in CMS Medicare drug coverage payments. According to the Medicare website, individuals with higher income generally pay more on Medicare Part D premiums.  This seems to more based on how wealthy the individual on Medicare is. The reason for the larger amount of variation for unvaccinated individuals may be because there may be a significant number of individuals who are not on Medicare. 

## Final Chosen Model
After observing the various models that were ran in this assessment, it was concluded that LightGBM is the most suitable model to identify individuals hesitant to take the vaccine. Although it misclassifies slightly higher than the other proposed models, it does the best job of properly differentiating between the vaccinated and unvaccinated individuals. The high AUC perfectly displays the advantage of the LightGBM over the other models. Aside from the speed advantage over XGBoost, LightGBM has an option that gives the ability to directly accept categorical variables without performing one-hot-encoding. This prevents the dataset from becoming too sparse, and it slightly increases the accuracy of the model. 

## Conclusion
While both XGBoost and LightGBM provided good results in predicting unvaccinated individuals and each provided a list of variables based on their importance, it did not provide a sense of directionality for these respective variables. For example, the models would commonly list the variable of estimated age near the top of their respective importance tables, but would not indicate if age being higher or lower would result in a greater likelihood of a non-vaccinated result. As a result, the team decided to select the top 20 attributes from the models and run a standard logistic regression for the variables that were referenced more than once. 

In running a simple logistic regression with these variables, we get a list of the variables with their respective p-values and the parameter estimates that can give us a sense of the directionality for that respective variable. For example, the parameter estimates for cms_risk_adjustment_factor_a_amt, one of the two variables that showed for all four of our machine-learning models, was 0.14927831. This leads us to believe that the higher an individual’s Risk Adjustment Factor A Amount, the greater likelihood the individual is to not be vaccinated. The opposite can be said for estimated age, the other variable to be shown in all four of our previously considered models. The logistic regression resulted in a parameter estimate of -0.0194982 which supports the argument that the younger an individual is, the greater likelihood the individual is to be vaccine hesitant. 

![Logistic Regression Features](https://github.com/MAdesuyi94/Team-Residuals/blob/MAdesuyi94-patch-1/Images/Logistic%20Regression%20Features.PNG)


## Recommendations
Based on the directionality of the logistic regression and comparing them to the results of the feature importance tables created by the XGBoost and LightGBM machine learning models, we recommend the following for encouraging higher vaccination rates among Humana members. 

### 1.	Increase education and engagement campaigns to encourage COVID-19 vaccinations in Gulf State and Texas Regions.
The Gulf State and Texas Humana regions were often referenced as high on the feature importance tables from the machine learning models and have a positive variable estimate in the Logistic Regression. While we cannot accurately infer why these regions show significantly higher than average hesitancy to the COVID-19 vaccine (for example: political preferences, education rates, and other social indicators), the trends show that they are important variables to the model and that they significantly increase the probability of an individual being classified as unvaccinated.  

### 2.	For all regions, focus efforts on areas which a higher population of young adults. 
Age appeared repeatedly as an important feature in our machine learning models. Examining the boxplots of the distributions of age against vaccination status, in addition to the directionality provided by the logistic regression, showed that the younger population were more likely to classified as unvaccinated. Although younger people may feel less at risk from dying or becoming seriously ill with COVID-19, they are still able to transmit the virus to others and are an important influence in stopping the spread of the disease. 

### 3.	Focus outreach and education for the disadvantaged and underserved populations.
Throughout the list of important features seen across multiple machine learning models, there are a miscellany of variables that would seem to indicate the individual potentially belonging to a marginalized, disadvantaged, or under4served patient population. Variables like cms_risk_adjustment_factor_a_amt show that people with a higher health risk factor are more likely to be unvaccinated. Results for rwjf_uninsured_child_pct and rwjf_uninsured_adults_pct showed that as percentages of children and adults without health insurance increased, so too did the probability of vaccine hesitancy. cons_nwperadult showed that as the net worth per adult increased, the likelihood of that individual being classified as unvaccinated also decreased. These, in addition to other high-importance variables, are clear indicators that certain high-risk communities have less access to outreach, education, and engagement around the COVID-19 vaccine. 

### 4.	Focus efforts on additional data collection to gain a complete data profile of each patient.
There were a few variables within the dataset that required considerable amounts of imputation, which can lead to erroneous and inaccurate conclusions in the data. Other variables still had too many missing values and were required to be dropped from the dataset completely. The patient zip code, which should be known for every member had approximately 40% missing or were recorded as a zip code that does not exist in the US.  To better understand each of these patients on a deeper level, we would recommend that further efforts be made to develop a complete and full profile of each member, so that future data modeling can be as accurate and as practical as possible. 
