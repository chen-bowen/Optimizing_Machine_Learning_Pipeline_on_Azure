# Optimizing an ML Pipeline in Azure

## Overview

(This project is part of the Udacity Azure ML Nanodegree.)

In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model in Azure ecosystem.
This model is then compared to an Azure AutoML run on the same problem. This project serves as a prime example of running machine learning tasks on a well-known cloud service provider 

## Summary

In this project, we seek to use 20 different demographical and event-related features to predict whether the user will eventually subscribe a term deposit. We will use sklearn's logistic regression framework, and tune its hyperparameters with HyperDrive. We will also utilize Azure's autoML feature to find us the best model for the bank marketing prediction problem.

## Dataset Description

The dataset is from UCI's machine learning repository, describing attributes and outcomes of all customers (whether they eventually subscribe a term deposit). 

### Attribute Information:

Input variables:

**bank client data:**

1. age (numeric)
2. job : type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')
3. marital : marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)
4. education (categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')
5. default: has credit in default? (categorical: 'no','yes','unknown')
6. housing: has housing loan? (categorical: 'no','yes','unknown')
7. loan: has personal loan? (categorical: 'no','yes','unknown')

**related with the last contact of the current campaign:**

8. contact: contact communication type (categorical: 'cellular','telephone')
9. month: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')
10. day_of_week: last contact day of the week (categorical: 'mon','tue','wed','thu','fri')
11. duration: last contact duration, in seconds (numeric). 

Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.

**other attributes:**

12. campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
13. pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
14. previous: number of contacts performed before this campaign and for this client (numeric)
15. poutcome: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')

**social and economic context attributes**
16. emp.var.rate: employment variation rate. quarterly indicator (numeric)
17. cons.price.idx: consumer price index - monthly indicator (numeric)
18. cons.conf.idx: consumer confidence index - monthly indicator (numeric)
19. euribor3m: euribor 3 month rate - daily indicator (numeric)
20. nr.employed: number of employees - quarterly indicator (numeric)

**Output variable (desired target):**
21. y - has the client subscribed a term deposit? (binary: 'yes','no')

## Scikit-learn Pipeline Architecture

### HyperDrive architecture

The hypedrive pipeline is built with these order of sequence

1. create the experiment inside the workspace
2. configure the compute cluster
3. create an sklearn estimator by setting up the main functionalities in the `train.py` script (including data preprocessing and model fitting)
4. use the `train.py` script as the entry point, create the `ScriptRunConfig` object 
5. use `ScriptRunConfig` object as entry point, create the `HyperDriveConfig` object
6. submit the `HyperDriveConfig` object for execution

A diagram of the pipeline structure is shown as the following

![Screenshot](images/pipeline%20architecture.png)

**benefits of the bayesian parameter sampler**

Using the BayesianParameterSampling sampler gives us a smart way to try a large range of hyperparameters to achieve the best performance. Using BayesianParameterSampling achieves comparable, if not better, performances compared to the brute force grid search parameter sampler.

**benefits of the early stopping policy**

Using the Bandit stopping policy allow us to cancel the runs that are using hyperparameters that lead to really bad performances. This will save us valuable runtime and computing resources to avoid paying for runs we would not use.

**parameters tuned in HyperDrive**

The following 2 parameters for logistic regression are tuned using hyperdrive 

* **C** - controls the inverse of regulation strength. The higher this value is, the higher propensity of the model is prone to overfitting due to reduced regularization strength

* **max_iter** - controls the maximum number of iterations taken for the logistic regression to converge. The higher this value is, the higher propensity of the model is prone to overfitting 

## AutoML

The AutoML pipeline is built with these order of sequence

1. create dataset from TabularDataSet object
2. configure AutoML settings by creating the `AutoMLConfig` object
3. create a designated experiment in the workspace
4. submit the `AutoMLConfig` object for execution

AutoML generates a variety of different models ranging from linear models to tree based models to ensemble models. The best model is the voting ensemble models, generating 0.9482 weighted AUC metric.

**The best model**

Voting ensemble model is a majority model that improves the model's predictive power by combining multiple models's results by a weighted average. Azure's voting ensemble model combines multiple models tried by AutoML previously, which improves the performance overall.

**Parameters generated by the best model**

The voting ensemble model is an weighted average model of 4 different xgboost models and 2 different lightGBM models with different scaling and preprocessing steps. The weights of each model is [0.26666666666666666, 0.2, 0.2, 0.13333333333333333, 0.06666666666666667, 0.13333333333333333] respectively. For example, the first model to be ensembled is an xgboost model that is preprocessed by a standardized scaler, with a set of specific parameters from hyperparameter tuning. The prediction is weighted by 0.267 in the final prediction result. 

Specific model definitions are shown below.

```
[('21',
  Pipeline(memory=None,
           steps=[('standardscalerwrapper',
                   <azureml.automl.runtime.shared.model_wrappers.StandardScalerWrapper object at 0x7f3162217a58>),
                  ('xgboostclassifier',
                   XGBoostClassifier(base_score=0.5, booster='gbtree',
                                     colsample_bylevel=1, colsample_bynode=1,
                                     colsample_bytree=1, eta=0.5, gamma=0.1,
                                     learning_rate=0.1, max_delta_step=0,
                                     max_depth=6, max_leaves=0,
                                     min_child_weight=1, missing=nan,
                                     n_estimators=50, n_jobs=1, nthread=None,
                                     objective='reg:logistic', random_state=0,
                                     reg_alpha=2.1875, reg_lambda=0,
                                     scale_pos_weight=1, seed=None, silent=None,
                                     subsample=0.9, tree_method='auto',
                                     verbose=-10, verbosity=0))],
           verbose=False)),
 ('0',
  Pipeline(memory=None,
           steps=[('maxabsscaler', MaxAbsScaler(copy=True)),
                  ('lightgbmclassifier',
                   LightGBMClassifier(boosting_type='gbdt', class_weight=None,
                                      colsample_bytree=1.0,
                                      importance_type='split', learning_rate=0.1,
                                      max_depth=-1, min_child_samples=20,
                                      min_child_weight=0.001, min_split_gain=0.0,
                                      n_estimators=100, n_jobs=1, num_leaves=31,
                                      objective=None, random_state=None,
                                      reg_alpha=0.0, reg_lambda=0.0, silent=True,
                                      subsample=1.0, subsample_for_bin=200000,
                                      subsample_freq=0, verbose=-10))],
           verbose=False)),
 ('1',
  Pipeline(memory=None,
           steps=[('maxabsscaler', MaxAbsScaler(copy=True)),
                  ('xgboostclassifier',
                   XGBoostClassifier(base_score=0.5, booster='gbtree',
                                     colsample_bylevel=1, colsample_bynode=1,
                                     colsample_bytree=1, gamma=0,
                                     learning_rate=0.1, max_delta_step=0,
                                     max_depth=3, min_child_weight=1, missing=nan,
                                     n_estimators=100, n_jobs=1, nthread=None,
                                     objective='binary:logistic', random_state=0,
                                     reg_alpha=0, reg_lambda=1,
                                     scale_pos_weight=1, seed=None, silent=None,
                                     subsample=1, tree_method='auto', verbose=-10,
                                     verbosity=0))],
           verbose=False)),
 ('22',
  Pipeline(memory=None,
           steps=[('standardscalerwrapper',
                   <azureml.automl.runtime.shared.model_wrappers.StandardScalerWrapper object at 0x7f315a21e908>),
                  ('xgboostclassifier',
                   XGBoostClassifier(base_score=0.5, booster='gbtree',
                                     colsample_bylevel=1, colsample_bynode=1,
                                     colsample_bytree=0.7, eta=0.01, gamma=0,
                                     grow_policy='lossguide', learning_rate=0.1,
                                     max_bin=255, max_delta_step=0, max_depth=6,
                                     max_leaves=0, min_child_weight=1,
                                     missing=nan, n_estimators=400, n_jobs=1,
                                     nthread=None, objective='reg:logistic',
                                     random_state=0, reg_alpha=1.6666666666666667,
                                     reg_lambda=0, scale_pos_weight=1, seed=None,
                                     silent=None, subsample=0.7,
                                     tree_method='hist', verbose=-10,
                                     verbosity=0))],
           verbose=False)),
 ('23',
  Pipeline(memory=None,
           steps=[('robustscaler',
                   RobustScaler(copy=True, quantile_range=[10, 90],
                                with_centering=True, with_scaling=False)),
                  ('lightgbmclassifier',
                   LightGBMClassifier(boosting_type='gbdt', class_weight=None,
                                      colsample_bytree=0.3966666666666666,
                                      importance_type='split',
                                      learning_rate=0.0842121052631579,
                                      max_bin=220, max_depth=4,
                                      min_child_samples=1904, min_child_weight=5,
                                      min_split_gain=0.15789473684210525,
                                      n_estimators=400, n_jobs=1, num_leaves=179,
                                      objective=None, random_state=None,
                                      reg_alpha=0, reg_lambda=0.10526315789473684,
                                      silent=True, subsample=0.5942105263157895,
                                      subsample_for_bin=200000, subsample_freq=0,
                                      verbose=-10))],
           verbose=False)),
 ('24',
  Pipeline(memory=None,
           steps=[('standardscalerwrapper',
                   <azureml.automl.runtime.shared.model_wrappers.StandardScalerWrapper object at 0x7f315a224240>),
                  ('xgboostclassifier',
                   XGBoostClassifier(base_score=0.5, booster='gbtree',
                                     colsample_bylevel=1, colsample_bynode=1,
                                     colsample_bytree=1, eta=0.01, gamma=0.01,
                                     learning_rate=0.1, max_delta_step=0,
                                     max_depth=8, max_leaves=31,
                                     min_child_weight=1, missing=nan,
                                     n_estimators=10, n_jobs=1, nthread=None,
                                     objective='reg:logistic', random_state=0,
                                     reg_alpha=2.291666666666667,
                                     reg_lambda=2.3958333333333335,
                                     scale_pos_weight=1, seed=None, silent=None,
                                     subsample=0.5, tree_method='auto',
                                     verbose=-10, verbosity=0))],
           verbose=False))]
```

## Pipeline comparison

The best model returned by autoML is the VotingEnsemble algorithm that returns an accuracy of 0.9143, compare to the logistic regression returns an accuracy metric of 0.9124. Logistic regression seek to predict the target variable with a linear combination of all the input features, while the voting ensemble model is a weighted average prediction of non-linear tree based models. The superior performance is highly dependent on whether the underlying classification problem is a linear one. From the result, our problem is better solved with a non-linear model.

## Future work
Some more detailed feature engineering prior to feeding the model into autoML would probably improve the accuracy/AUC by a lot. Combining the power of feature engineering and autoML will allow us to automatically try a lot of different models on already nicely engineered features. ALso using cross validation when performing autoML will effectively reduce overfitting, which could potentially improve the performance of the best model as well.
