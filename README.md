# Humana -Mays Case Competition – 2021
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

## Data Visualization
First, each variable was examined based on its distribution. Distribution plots were created for each numeric variable and assessed for normality, also noting any specific outliers. 

Next, box plots were created for the numeric variables to search for any visual relationships against the target variable. Specifically, we were looking for features that showed differences in the distributions and average values when comparing the two target classes.
 

Count plots were evaluated for categorical features, searching for categories that show significant differences between the target class.
 

Density plots were also evaluated to examine the differences in distributions from vaccinated to unvaccinated individuals. 
 

