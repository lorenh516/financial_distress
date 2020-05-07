# financial_distress
Machine Learning Pipeline  + Exploration of Credit Data

## Task Description:

Predict which people will experience financial distress in the next two years 
based on:
* 10 numeric variables
* an identification number
* one categorical geographic variable (zip code). 

The outcome variable (label) in the data is `SeriousDlqin2yrs`.

## Contents

data: directory containing dataset CSV file containing (modified version of 
data from https://www.kaggle.com/c/GiveMeSomeCredit) and .xls data dictionary

credit_util.py: assignment-specific utility functions 

finan_distress.ipynb: ipython notebook calling pipeline functions to predict 
	financial distress 

ml_pipeline_lch.py: machine learning pipeline functions for reading and 
	pre-processing data

ml_explore.py: pipeline functions for data exploration

ml_modeling.p: machine learning pipeline functions for decision tree model 
	building

tree.dot: visualization of decision tree model


#### Repository requirements:
```{python}
pip install -r requirements.txt
```
