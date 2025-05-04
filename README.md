![](UTA-DataScience-Logo.png)

# Predicting Metastatic Cancer Diagnosis

* This repository holds an attempt to apply machine learning techniques and models to metastatic cancer diagnosis to predict id patients recieved a cancer diagnosis withing 90 days of screening using data from
"WiDS Datathon 2024 Challenge #1" Kaggle challenge [(https://www.kaggle.com/competitions/widsdatathon2024-challenge1)]. 

## Overview

* The task, as defined by the Kaggle challenge is to develop a model to predict if patients recieved a metatstatic cancer diagnosis within 90 days of screening. This repository approaches this problem as a binary classification task, using 2 different models, Random Forest Classifier and a variant of a gradient boost model called, CatBoost were compared against each other. CatBoost was the best model for the task as it was able to predict whether a patient was diagnosed with metastatic cancer within 90 days of screening scored at ~81% accuracy. At the time of this writing, the best performance on the Kaggle leaderboards of this metric is 82%.

## Summary of Work Done

### Data

* Data:
  * Type: Binary Classification
    * Input: CSV file: train.csv, test.csv; described patient information (eg. age, gender, educational background, etc)
    * Output: sucess or failure based on whether or not pateint received a metastatic cancer daignosis within 90 days of screening -> target col = 'DiagPeriodL90D'
  * Size: Original training and testing datasets together was 16 MB (training: 12,906 rows & 83 features; test: 5792 rows & 82 features). After cleaning and proper preprocessing both datasets together was about 36 MB.
  * Instances (Train, Test, Validation Split): training: 12906, testing: 5792.

#### Preprocessing / Clean up

* Dropped features that only had one unique values as they lacked predictive power and variability, and dropped redundant features.
* During visualization and correlation, patient age seemed to be the most trusted feature so missing values in the features, 'patient_race', 'bmi', and 'payer_type' were imputed by the average of the age group.
* The rest of the missing values in other features that were below 2%, were imputed with the most frequent value (mode).
* No feature scaling (standardization or normalization) was done.
* The data sets were split into two separate versions of the cleaned dataset. One included all of the above imputations and removal of the columns (2) dataset inherited the changes from the 1st dataset but included one-hot encoding. 

#### Data Visualization
The following visualizations compare the success/failure for the metastatic cancer diagnosis within 90 days of screening for each feature.

![Screenshot 2024-05-05 180917](https://github.com/HyItsAngela/DATA3402.PROJECT/assets/143844332/4cd99122-7989-451b-b23c-30c2cd80e0b7)

Visual of a select few categorical features. Not all features were visualized to avoid overcrowding.

![Screenshot 2024-05-05 181137](https://github.com/HyItsAngela/DATA3402.PROJECT/assets/143844332/4919c009-0464-4869-8705-a6972fab8d6c)

Zoomed in example distributions for a couple of numerical features.

![Screenshot 2024-05-03 113355](https://github.com/HyItsAngela/DATA3402.PROJECT/assets/143844332/05641d3c-308a-41a5-bda0-bbac01102c8d)

Overview distributions of more numerical features.

### Problem Formulation

* Train information about demographics, diagnosis and treatment options, insurance and more with machine learning to provide a better view about aspects that may contribute to health equity.
  * Models
    * RandomForest; chosen for it's ease and flexibility and hence used as a base model for comparison.
    * Catboost; chosen for it's built-in methods, predictive power and great results without the need for parameter tuning, and robustness.
  * No in-depth fine-tuning or optimization to the models such as hypyerparameters, feature importance or cross validation were done. 

### Training

* Describe the training:
  * Training was done on a Surface Pro 9 using Python via jupyter notebook.
  * Training did not take long to process, with the longest training time to be approximately a minute.
  * Concluded training when results were satisfactory and plenty of evaluation metrics for comparison observed fairly decent results.

### Performance Comparison

* Key performance metrics were imported from sklearn and consist of:
  * log_loss().
  * classification_report().
  * accuracy_score().
  * roc_auc_score().
  * roc_curve().
  * auc().

![Screenshot 2024-05-05 183147](https://github.com/HyItsAngela/DATA3402.PROJECT/assets/143844332/20fcdcab-79aa-4536-a5c9-0684119ac479)

Table of metrics to compare and contrast the evaluations of the models. CatBoost model exhibited 
 
![Screenshot 2024-05-05 183044](https://github.com/HyItsAngela/DATA3402.PROJECT/assets/143844332/35a89034-6806-431e-b588-992c2c167ae3)

ROC curve and AUC measurement comparisons for models, RandomForest and CatBoost. The higher the AUC the better the model, CatBoost has a higher AUC (0.80).

### Conclusions

* CatBoost worked better than RandomForest. Both models could have yeilded better accuracy if time was taken to perform hyper tuning, training with important features, and using cross validation techniques, but as seen from it's ~81% accuracy the model did quite well with it's default training techniques and shines through with it's robustness and preprocessing techniques they're known for. 

### Future Work

* For future personal exploration, I would definitely like to dive deeper into feature importance, hypertuning parameters, and cross validation to see what features affect not only the model but the diagnosis and treatment of the patients to really study and understand healthcare equity.
* Future studies for others can dive into the environmental pollutants, and other features that patients are exposed to, to study correlations and possibly causations with the positive occurrences of cancer screening.

## How to reproduce results

* The notebooks are well organized and include further explanation but for a summary:
* Download the original data files ('train.csv', 'test.csv') from Kaggle or directly through the current repository along with the processed data files.
* Install the necessary libraries
* Run the notebooks attached
* As long as a platform that can provide Python, such as Collab, Anaconda, etc, is used, results can be replicated.

### Overview of files in repository

* The repository includes 11 files in total.
  * Initial Look.py: Introduces the analyst to the data by looking and understanding the features, missing values, and class imbalances
  * Data Preparation.ipynb: Applies the understaning from Intial Look.ipynb and begins to understand correlations and start cleaning, preprocessing, and visualizing.
  * Machine.ipynb: Trains, predicts, evaluates, and visualizes the data using machine learning models.
  * Kaggle Tabular Data.ipynb: Contains the outline for the project.
  * submission.csv: Submission file for the Kaggle challenge that includes the final predictions of the model.
  * cleaned_test_df.csv: Cleaned test dataset that includes dropped columns and imputations of missing values.
  * cleaned_train_df.csv: Cleaned train dataset that includes dropped columns and imputations of missing values.
  * prep_test_df.csv: Preprocessed test dataset that inherits changes made from 'cleaned_test_df.csv' but adds one-hot encoding.
  * prep_train_df.csv: Preprocessed train dataset that inherits changes made from 'cleaned_train_df.csv' but adds one-hot encoding.
  * training.csv: Official and original training dataset that was provided from Kaggle
  * test.csv: Official and original test dataset that was provided from Kaggle


### Software Setup
* Required Packages:
  * Numpy
  * Pandas
  * Sklearn
  * Seaborn
  * Matplotlib.pyplot
  * Math
  * Catboost
  * Scipy
  * Tabulate
* Installlation Proccess:
  * Installed through Linux subsystem for Windows
  * Installed via Ubuntu
  * pip3 install numpy
  * pip3 install pandas
  * pip3 install -U scikit-learn
  * pip! install catboost

### Data

* Data can be downloaded through the official Kaggle website through the link stated above. Or through Kaggle's API interface.

### Training

* Models can be trained by first splitting the testing dataset into two datasets to be trained and validated. Choose the model you wish to train and fit the data and validation variables. Look below in citations to research official websites to find parameters of the model functions to tune to your liking.

#### Performance Evaluation

* Evaluation metrics are imported such as the log loss, accuracy score, classification score. The ROC curve and AUC measurement were also imported and then placed into a function for comparison of multiple models.
* Run the notebooks.


## Citations

* Official CatBoost website; used to learn about the CatBoost model and parameters it has to offer: https://catboost.ai/en/docs/concepts/python-reference_catboostclassifier_eval-metrics
* Official SciKit-Learn website; used to learn about RandomForest and other potential models: https://scikit-learn.org/stable/supervised_learning.html#supervised-learning
* Fellow participant in Kaggle challenge; incorporated similar imputation technique for 'patient_race', 'bmi', and 'payer_type': https://www.kaggle.com/code/khueluu/wids2024-acwomen-catboost#3.-Model







