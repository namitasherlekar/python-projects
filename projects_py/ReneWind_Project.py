#!/usr/bin/env python
# coding: utf-8

# # ReneWind Project
# 
# Renewable energy sources play an increasingly important role in the global energy mix, as the effort to reduce the environmental impact of energy production increases.
# 
# Out of all the renewable energy alternatives, wind energy is one of the most developed technologies worldwide. The U.S Department of Energy has put together a guide to achieving operational efficiency using predictive maintenance practices.
# 
# Predictive maintenance uses sensor information and analysis methods to measure and predict degradation and future component capability. The idea behind predictive maintenance is that failure patterns are predictable and if component failure can be predicted accurately and the component is replaced before it fails, the costs of operation and maintenance will be much lower.
# 
# The sensors fitted across different machines involved in the process of energy generation collect data related to various environmental factors (temperature, humidity, wind speed, etc.) and additional features related to various parts of the wind turbine (gearbox, tower, blades, break, etc.). 
# 
# 
# 
# ## Objective
# “ReneWind” is a company working on improving the machinery/processes involved in the production of wind energy using machine learning and has collected data of generator failure of wind turbines using sensors. They have shared a ciphered version of the data, as the data collected through sensors is confidential (the type of data collected varies with companies). Data has 40 predictors, 20000 observations in the training set and 5000 in the test set.
# 
# The objective is to build various classification models, tune them, and find the best one that will help identify failures so that the generators could be repaired before failing/breaking to reduce the overall maintenance cost. 
# The nature of predictions made by the classification model will translate as follows:
# 
# - True positives (TP) are failures correctly predicted by the model. These will result in repairing costs.
# - False negatives (FN) are real failures where there is no detection by the model. These will result in replacement costs.
# - False positives (FP) are detections where there is no failure. These will result in inspection costs.
# 
# It is given that the cost of repairing a generator is much less than the cost of replacing it, and the cost of inspection is less than the cost of repair.
# 
# “1” in the target variables should be considered as “failure” and “0” represents “No failure”.
# 
# ## Data Description
# - The data provided is a transformed version of original data which was collected using sensors.
# - Train.csv - To be used for training and tuning of models. 
# - Test.csv - To be used only for testing the performance of the final best model.
# - Both the datasets consist of 40 predictor variables and 1 target variable
# 
# ## Importing libraries

# In[1]:


# suppress all warnings
import warnings
warnings.filterwarnings("ignore")

#import libraries needed for data manipulation
import pandas as pd
import numpy as np

pd.set_option("display.float_format", lambda x: "%.3f" % x)

#import libraries needed for data visualization

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

# using statsmodels to build our model
import statsmodels.stats.api as sms
import statsmodels.api as sm

# unlimited number of displayed columns and rows
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

# split the data into random train and test subsets
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score

# Libraries different ensemble classifiers
from sklearn.ensemble import (
    BaggingClassifier,
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    StackingClassifier,
)

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

# Libraries to get different metric scores
from sklearn import metrics
from sklearn.metrics import (
    f1_score,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    plot_confusion_matrix,
)

# To be used for data scaling and one hot encoding
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder

# To impute missing values
from sklearn.impute import SimpleImputer

# To oversample and undersample data
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# To do hyperparameter tuning
from sklearn.model_selection import RandomizedSearchCV

# To be used for creating pipelines and personalizing them
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# To tune different models
from sklearn.model_selection import GridSearchCV


# In[2]:


#import datasets named 'Train.csv' and 'Test.csv'

data = pd.read_csv('Train.csv')
df_test = pd.read_csv('Test.csv')

# read first five rows of the training dataset

data.head()


# In[3]:


# read last five rows of dataset

data.tail()


# In[4]:


#check shape of training dataset

data.shape


# In[5]:


#check shape of test dataset

df_test.shape


# ## Data Overview
# 
# - Observations
# - Sanity checks

# In[6]:


df = data.copy()


# In[7]:


test = df_test.copy()


# In[8]:


df.info()


# In[9]:


df.isnull().sum()


# In[10]:


test.isnull().sum()


# In[11]:


df.duplicated().sum()


# In[12]:


test.duplicated().sum()


# In[13]:


df.describe().T


# ### Observations
# 
# - Train set
#     - 20000 rows
# - Test set
#     - 5000 rows
# - 41 variables: 40 predictors, 1 target
# - First two predictor variables have a few missing values
# - No duplicate values
# - Not much variation in means across predictor variables
# - Some ranges are surpsingly large
# - Target variable ranges from 0-1, averaging at 0.056
# 
# ## EDA
# ### Univariate Analysis

# In[14]:


# define a function to plot a boxplot and a histogram along the same scale

def histbox(data, feature, figsize=(12, 7), kde=False, bins=None):
    """
    Boxplot and histogram combined
    data: dataframe
    feature: dataframe column
    figsize: size of figure (default (12,7))
    kde: whether to show the density curve (default False)
    bins: number of bins for histogram (default None)
    """
    f2, (box, hist) = plt.subplots(
        nrows=2,                                            # Number of rows of the subplot grid = 2
                                                                # boxplot first then histogram created below 
        sharex=True,                                        # x-axis same among all subplots
        gridspec_kw={"height_ratios": (0.25, 0.75)},        # boxplot 1/3 height of histogram
        figsize=figsize,                                    # figsize defined above as (12, 7)
    )  
    # defining boxplot inside function, so when using it say histbox(df, 'cost'), df: data and cost: feature
    
    sns.boxplot(
        data=data, x=feature, ax=box, showmeans=True, color="chocolate"
    )  # showmeans makes mean val on boxplot have star, ax = 
    sns.histplot(
        data=data, x=feature, kde=kde, ax=hist, bins=bins, color = "darkgreen"
    ) if bins else sns.histplot(
        data=data, x=feature, kde=kde, ax=hist, color = "darkgreen"
    )  # For histogram if there are bins in potential graph 
    
    # add vertical line in histogram for mean and median
    hist.axvline( 
        data[feature].mean(), color="purple", linestyle="--"
    )  # Add mean to the histogram
    hist.axvline(
        data[feature].median(), color="black", linestyle="-"
    )  # Add median to the histogram


# In[15]:


# plot histogram/boxplot for all variables

for feature in data.columns:
    histbox(df, feature, figsize=(12, 7), kde=False, bins=None)


# ### Observations
# 
# - The 40 predictor variables have very similar distributions, with the ranges on average not exceeding 10. 
# - Many outliers on both the lower and upper ends of the predictor variable distributions.
# - We will now analyze the target variable separately to identify the pattern: above shows multimodal.

# In[16]:


df["Target"].value_counts()


# In[17]:


test["Target"].value_counts()


# ### Observations
# 
# - “1” in the target variables should be considered as “failure” and “0” represents “No failure”.
# - Test set has a slightly higher proportion of failures (56.4% vs 55.5%)
# 
# ## Data Pre-processing

# In[18]:


X = df.drop(["Target"], axis=1)
y = df["Target"]

X_test = test.drop(["Target"], axis=1)
y_test = test["Target"]


# In[19]:


# splitting the data in 70:30 ratio for train to test data

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=1)

print(X_train.shape, X_val.shape, X_test.shape)


# In[20]:


imputer = SimpleImputer(strategy="median")


# In[21]:


# Fit and transform the train data
X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
X_val = pd.DataFrame(imputer.fit_transform(X_val), columns=X_val.columns)
X_test = pd.DataFrame(imputer.fit_transform(X_test), columns=X_test.columns)


# In[22]:


# check for missing values 

print(X_train.isnull().sum())
print("-" * 30)
print(X_val.isnull().sum())
print("-" * 30)
print(X_test.isnull().sum())


# ## Model Building
# 
# ### Model evaluation criterion
# 
# The nature of predictions made by the classification model will translate as follows:
# 
# - True positives (TP) are failures correctly predicted by the model.
# - False negatives (FN) are real failures in a generator where there is no detection by model. 
# - False positives (FP) are failure detections in a generator where there is no failure.
# 
# ### Which metric to optimize?
# 
# * We need to choose the metric which will ensure that the maximum number of generator failures are predicted correctly by the model.
# * We would want Recall to be maximized as greater the Recall, the higher the chances of minimizing false negatives.
# * We want to minimize false negatives because if a model predicts that a machine will have no failure when there will be a failure, it will increase the maintenance cost.
# 
# **Let's define a function to output different metrics (including recall) on the train and test set and a function to show confusion matrix so that we do not have to use the same code repetitively while evaluating models.**

# In[23]:


# defining a function to compute different metrics to check performance of a classification model built using sklearn
def model_performance_classification_sklearn(model, predictors, target):
    """
    Function to compute different metrics to check classification model performance

    model: classifier
    predictors: independent variables
    target: dependent variable
    """

    # predicting using the independent variables
    pred = model.predict(predictors)

    acc = accuracy_score(target, pred)  # to compute Accuracy
    recall = recall_score(target, pred)  # to compute Recall
    precision = precision_score(target, pred)  # to compute Precision
    f1 = f1_score(target, pred)  # to compute F1-score

    # creating a dataframe of metrics
    df_perf = pd.DataFrame(
        {
            "Accuracy": acc,
            "Recall": recall,
            "Precision": precision,
            "F1": f1
            
        },
        index=[0],
    )

    return df_perf


# In[24]:


def confusion_matrix_sklearn(model, predictors, target):
    """
    To plot the confusion_matrix with percentages

    model: classifier
    predictors: independent variables
    target: dependent variable
    """
    y_pred = model.predict(predictors)
    cm = confusion_matrix(target, y_pred)
    labels = np.asarray(
        [
            ["{0:0.0f}".format(item) + "\n{0:.2%}".format(item / cm.flatten().sum())]
            for item in cm.flatten()
        ]
    ).reshape(2, 2)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=labels, fmt="")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")


# ### Defining scorer to be used for cross-validation and hyperparameter tuning
# 
# - We want to reduce false negatives and will try to maximize "Recall".
# - To maximize Recall, we can use Recall as a **scorer** in cross-validation and hyperparameter tuning.

# In[25]:


# Type of scoring used to compare parameter combinations
scorer = metrics.make_scorer(metrics.recall_score)


# ### Model Building on original data

# In[26]:


models = []  # Empty list to store all the models

# Appending models into the list
models.append(("Bagging", BaggingClassifier(random_state=1)))
models.append(("Random forest", RandomForestClassifier(random_state=1)))
models.append(("GBM", GradientBoostingClassifier(random_state=1)))
models.append(("Adaboost", AdaBoostClassifier(random_state=1)))
models.append(("Logistic regression", LogisticRegression(random_state=1)))
models.append(("dtree", DecisionTreeClassifier(random_state=1)))

results1 = []  # Empty list to store all model's CV scores
names = []  # Empty list to store name of the models


# loop through all models to get the mean cross validated score
print("\n" "Cross-Validation Performance:" "\n")

for name, model in models:
    kfold = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=1
    )  # Setting number of splits equal to 5
    cv_result = cross_val_score(
        estimator=model, X=X_train, y=y_train, scoring=scorer, cv=kfold
    )
    results1.append(cv_result)
    names.append(name)
    print("{}: {}".format(name, cv_result.mean()))

print("\n" "Validation Performance:" "\n")

for name, model in models:
    model.fit(X_train, y_train)
    scores = recall_score(y_val, model.predict(X_val))
    print("{}: {}".format(name, scores))


# In[27]:


# Plotting boxplots for CV scores of all models defined above
fig = plt.figure(figsize=(10, 7))

fig.suptitle("Algorithm Comparison")
ax = fig.add_subplot(111)

plt.boxplot(results1)
ax.set_xticklabels(names)

plt.show()


# ### Model Building on oversampled data

# In[28]:


# Synthetic Minority Over Sampling Technique
sm = SMOTE(sampling_strategy=1, k_neighbors=5, random_state=1)
X_train_over, y_train_over = sm.fit_resample(X_train, y_train)

print("Before OverSampling, counts of label '1': {}".format(sum(y_train == 1)))
print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train == 0)))

print("After OverSampling, counts of label '1': {}".format(sum(y_train_over == 1)))
print("After OverSampling, counts of label '0': {} \n".format(sum(y_train_over == 0)))

print("After OverSampling, the shape of train_X: {}".format(X_train_over.shape))
print("After OverSampling, the shape of train_y: {} \n".format(y_train_over.shape))


# In[29]:


models_over = []  # Empty list to store all the models

# Appending models into the list
models_over.append(("Bagging", BaggingClassifier(random_state=1)))
models_over.append(("Random forest", RandomForestClassifier(random_state=1)))
models_over.append(("GBM", GradientBoostingClassifier(random_state=1)))
models_over.append(("Adaboost", AdaBoostClassifier(random_state=1)))
models_over.append(("Logistic regression", LogisticRegression(random_state=1)))
models_over.append(("dtree", DecisionTreeClassifier(random_state=1)))

results2 = []  # Empty list to store all model's CV scores
names = []  # Empty list to store name of the models


# loop through all models to get the mean cross validated score
print("\n" "Cross-Validation Performance:" "\n")

for name, model in models_over:
    kfold = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=1
    )  # Setting number of splits equal to 5
    cv_result = cross_val_score(
        estimator=model, X=X_train_over, y=y_train_over, scoring=scorer, cv=kfold
    )
    results2.append(cv_result)
    names.append(name)
    print("{}: {}".format(name, cv_result.mean()))

print("\n" "Validation Performance:" "\n")

for name, model in models_over:
    model.fit(X_train_over, y_train_over)
    scores = recall_score(y_val, model.predict(X_val))
    print("{}: {}".format(name, scores))


# In[30]:


# Plotting boxplots for CV scores of all models defined above (oversampled)
fig = plt.figure(figsize=(10, 7))

fig.suptitle("Algorithm Comparison")
ax = fig.add_subplot(111)

plt.boxplot(results2)
ax.set_xticklabels(names)

plt.show()


# ### Model Building on undersampled data

# In[31]:


rus = RandomUnderSampler(random_state=1, sampling_strategy=1)
X_train_un, y_train_un = rus.fit_resample(X_train, y_train)


print("Before UnderSampling, counts of label '1': {}".format(sum(y_train == 1)))
print("Before UnderSampling, counts of label '0': {} \n".format(sum(y_train == 0)))

print("After UnderSampling, counts of label '1': {}".format(sum(y_train_un == 1)))
print("After UnderSampling, counts of label '0': {} \n".format(sum(y_train_un == 0)))

print("After UnderSampling, the shape of train_X: {}".format(X_train_un.shape))
print("After UnderSampling, the shape of train_y: {} \n".format(y_train_un.shape))


# In[32]:


models_un = []  # Empty list to store all the models

# Appending models into the list
models_un.append(("Bagging", BaggingClassifier(random_state=1)))
models_un.append(("Random forest", RandomForestClassifier(random_state=1)))
models_un.append(("GBM", GradientBoostingClassifier(random_state=1)))
models_un.append(("Adaboost", AdaBoostClassifier(random_state=1)))
models_un.append(("Logistic regression", LogisticRegression(random_state=1)))
models_un.append(("dtree", DecisionTreeClassifier(random_state=1)))

results3 = []  # Empty list to store all model's CV scores
names = []  # Empty list to store name of the models


# loop through all models to get the mean cross validated score
print("\n" "Cross-Validation Performance:" "\n")

for name, model in models_un:
    kfold = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=1
    )  # Setting number of splits equal to 5
    cv_result = cross_val_score(
        estimator=model, X=X_train_un, y=y_train_un, scoring=scorer, cv=kfold
    )
    results3.append(cv_result)
    names.append(name)
    print("{}: {}".format(name, cv_result.mean()))

print("\n" "Validation Performance:" "\n")

for name, model in models_un:
    model.fit(X_train_un, y_train_un)
    scores = recall_score(y_val, model.predict(X_val))
    print("{}: {}".format(name, scores))


# In[116]:


# Plotting boxplots for CV scores of all models defined above (undersampled)
fig = plt.figure(figsize=(10, 7))

fig.suptitle("Algorithm Comparison")
ax = fig.add_subplot(111)

plt.boxplot(results3)
ax.set_xticklabels(names)

plt.show()


# **After looking at performance of all the models, let's decide which models can further improve with hyperparameter tuning.**
# 
# ## Hyperparameter Tuning
# ### Tuning Bagging: Original

# In[57]:


# defining model
Model = BaggingClassifier(random_state=1)

# Parameter grid to pass in RandomSearchCV
param_grid = {
    'max_samples': [0.8,0.9,1],
    'max_features': [0.7,0.8,0.9],
    'n_estimators' : [30,50,70], }

#Calling RandomizedSearchCV
randomized_cv = RandomizedSearchCV(estimator=Model, param_distributions=param_grid, n_iter=50, n_jobs = -1, scoring=scorer, cv=5, random_state=1)

#Fitting parameters in RandomizedSearchCV
randomized_cv.fit(X_train,y_train)

print("Best parameters are {} with CV score={}:" .format(randomized_cv.best_params_,randomized_cv.best_score_))


# ### Tuning Bagging: Oversampled 

# In[58]:


# defining model
Model = BaggingClassifier(random_state=1)

# Parameter grid to pass in RandomSearchCV
param_grid = {
    'max_samples': [0.8,0.9,1],
    'max_features': [0.7,0.8,0.9],
    'n_estimators' : [30,50,70], }

#Calling RandomizedSearchCV
randomized_cv = RandomizedSearchCV(estimator=Model, param_distributions=param_grid, n_iter=50, n_jobs = -1, scoring=scorer, cv=5, random_state=1)

#Fitting parameters in RandomizedSearchCV
randomized_cv.fit(X_train_over,y_train_over)

print("Best parameters are {} with CV score={}:" .format(randomized_cv.best_params_,randomized_cv.best_score_))


# ### Tuning Bagging: Undersampled

# In[59]:


# defining model
Model = BaggingClassifier(random_state=1)

# Parameter grid to pass in RandomSearchCV
param_grid = {
    'max_samples': [0.8,0.9,1],
    'max_features': [0.7,0.8,0.9],
    'n_estimators' : [30,50,70], }

#Calling RandomizedSearchCV
randomized_cv = RandomizedSearchCV(estimator=Model, param_distributions=param_grid, n_iter=50, n_jobs = -1, scoring=scorer, cv=5, random_state=1)

#Fitting parameters in RandomizedSearchCV
randomized_cv.fit(X_train_un,y_train_un)

print("Best parameters are {} with CV score={}:" .format(randomized_cv.best_params_,randomized_cv.best_score_))


# ### Tuning Decision Tree: Original

# In[41]:


# defining model
Model = DecisionTreeClassifier(random_state=1)

# Parameter grid to pass in RandomSearchCV
param_grid = {'max_depth': np.arange(2,6),
              'min_samples_leaf': [1, 4, 7], 
              'max_leaf_nodes' : [10,15],
              'min_impurity_decrease': [0.0001,0.001] }

#Calling RandomizedSearchCV
randomized_cv = RandomizedSearchCV(estimator=Model, param_distributions=param_grid, n_iter=10, n_jobs = -1, scoring=scorer, cv=5, random_state=1)

#Fitting parameters in RandomizedSearchCV
randomized_cv.fit(X_train,y_train)

print("Best parameters are {} with CV score={}:" .format(randomized_cv.best_params_,randomized_cv.best_score_))


# ### Tuning Decision Tree: Oversampled

# In[47]:


# defining model
Model = DecisionTreeClassifier(random_state=1)

# Parameter grid to pass in RandomSearchCV
param_grid = {'max_depth': np.arange(2,6),
              'min_samples_leaf': [1, 4, 7], 
              'max_leaf_nodes' : [10,15],
              'min_impurity_decrease': [0.0001,0.001] }

#Calling RandomizedSearchCV
randomized_cv = RandomizedSearchCV(estimator=Model, param_distributions=param_grid, n_iter=10, n_jobs = -1, scoring=scorer, cv=5, random_state=1)

#Fitting parameters in RandomizedSearchCV
randomized_cv.fit(X_train_over,y_train_over)

print("Best parameters are {} with CV score={}:" .format(randomized_cv.best_params_,randomized_cv.best_score_))


# ### Tuning Decision Tree: Undersampled

# In[43]:


# defining model
Model = DecisionTreeClassifier(random_state=1)

# Parameter grid to pass in RandomSearchCV
param_grid = {'max_depth': np.arange(2,20),
              'min_samples_leaf': [1, 2, 5, 7], 
              'max_leaf_nodes' : [5, 10,15],
              'min_impurity_decrease': [0.0001,0.001] }

#Calling RandomizedSearchCV
randomized_cv = RandomizedSearchCV(estimator=Model, param_distributions=param_grid, n_iter=10, n_jobs = -1, scoring=scorer, cv=5, random_state=1)

#Fitting parameters in RandomizedSearchCV
randomized_cv.fit(X_train_un,y_train_un)

print("Best parameters are {} with CV score={}:" .format(randomized_cv.best_params_,randomized_cv.best_score_))


# ### Tuning Random Forest: Original

# In[44]:


# defining model
Model = RandomForestClassifier(random_state=1)

# Parameter grid to pass in RandomSearchCV
param_grid = {
    "n_estimators": [200,250,300],
    "min_samples_leaf": np.arange(1, 4),
    "max_features": [np.arange(0.3, 0.6, 0.1),'sqrt'],
    "max_samples": np.arange(0.4, 0.7, 0.1)}


#Calling RandomizedSearchCV
randomized_cv = RandomizedSearchCV(estimator=Model, param_distributions=param_grid, n_iter=50, n_jobs = -1, scoring=scorer, cv=5, random_state=1)

#Fitting parameters in RandomizedSearchCV
randomized_cv.fit(X_train, y_train)

print("Best parameters are {} with CV score={}:" .format(randomized_cv.best_params_,randomized_cv.best_score_))


# ### Tuning Random Forest: Oversampled

# In[45]:


# defining model
Model = RandomForestClassifier(random_state=1)

# Parameter grid to pass in RandomSearchCV
param_grid = {
    "n_estimators": [200,250,300],
    "min_samples_leaf": np.arange(1, 4),
    "max_features": [np.arange(0.3, 0.6, 0.1),'sqrt'],
    "max_samples": np.arange(0.4, 0.7, 0.1)}


#Calling RandomizedSearchCV
randomized_cv = RandomizedSearchCV(estimator=Model, param_distributions=param_grid, n_iter=50, n_jobs = -1, scoring=scorer, cv=5, random_state=1)

#Fitting parameters in RandomizedSearchCV
randomized_cv.fit(X_train_over, y_train_over)

print("Best parameters are {} with CV score={}:" .format(randomized_cv.best_params_,randomized_cv.best_score_))


# ### Tuning Random Forest: Undersampled

# In[46]:


# defining model
Model = RandomForestClassifier(random_state=1)

# Parameter grid to pass in RandomSearchCV
param_grid = {
    "n_estimators": [200,250,300],
    "min_samples_leaf": np.arange(1, 4),
    "max_features": [np.arange(0.3, 0.6, 0.1),'sqrt'],
    "max_samples": np.arange(0.4, 0.7, 0.1)}


#Calling RandomizedSearchCV
randomized_cv = RandomizedSearchCV(estimator=Model, param_distributions=param_grid, n_iter=50, n_jobs = -1, scoring=scorer, cv=5, random_state=1)

#Fitting parameters in RandomizedSearchCV
randomized_cv.fit(X_train_un, y_train_un)

print("Best parameters are {} with CV score={}:" .format(randomized_cv.best_params_,randomized_cv.best_score_))


# ### Tuning Gradient Boosting: Original

# In[48]:


# defining model
Model = GradientBoostingClassifier(random_state=1)

#Parameter grid to pass in RandomSearchCV
param_grid={
    "n_estimators": np.arange(100,150,25), 
    "learning_rate": [0.2, 0.05, 1], 
    "subsample":[0.5,0.7], 
    "max_features":[0.5,0.7]}

#Calling RandomizedSearchCV
randomized_cv = RandomizedSearchCV(estimator=Model, param_distributions=param_grid, scoring=scorer, n_iter=50, n_jobs = -1, cv=5, random_state=1)

#Fitting parameters in RandomizedSearchCV
randomized_cv.fit(X_train, y_train)

print("Best parameters are {} with CV score={}:" .format(randomized_cv.best_params_,randomized_cv.best_score_))


# ### Tuning Gradient Boosting: Oversampled

# In[49]:


# defining model
Model = GradientBoostingClassifier(random_state=1)

#Parameter grid to pass in RandomSearchCV
param_grid={
    "n_estimators": np.arange(100,150,25), 
    "learning_rate": [0.2, 0.05, 1], 
    "subsample":[0.5,0.7], 
    "max_features":[0.5,0.7]}

#Calling RandomizedSearchCV
randomized_cv = RandomizedSearchCV(estimator=Model, param_distributions=param_grid, scoring=scorer, n_iter=50, n_jobs = -1, cv=5, random_state=1)

#Fitting parameters in RandomizedSearchCV
randomized_cv.fit(X_train_over, y_train_over)

print("Best parameters are {} with CV score={}:" .format(randomized_cv.best_params_,randomized_cv.best_score_))


# ### Tuning Gradient Boosting: Undersampled

# In[50]:


# defining model
Model = GradientBoostingClassifier(random_state=1)

#Parameter grid to pass in RandomSearchCV
param_grid={
    "n_estimators": np.arange(100,150,25), 
    "learning_rate": [0.2, 0.05, 1], 
    "subsample":[0.5,0.7], 
    "max_features":[0.5,0.7]}

#Calling RandomizedSearchCV
randomized_cv = RandomizedSearchCV(estimator=Model, param_distributions=param_grid, scoring=scorer, n_iter=50, n_jobs = -1, cv=5, random_state=1)

#Fitting parameters in RandomizedSearchCV
randomized_cv.fit(X_train_un, y_train_un)

print("Best parameters are {} with CV score={}:" .format(randomized_cv.best_params_,randomized_cv.best_score_))


# ### Tuning Adaboost: Original

# In[51]:


# defining model
Model = AdaBoostClassifier(random_state=1)

# Parameter grid to pass in RandomSearchCV
param_grid = {
    "n_estimators": [100, 150, 200],
    "learning_rate": [0.2, 0.05],
    "base_estimator": [DecisionTreeClassifier(max_depth=1, random_state=1), DecisionTreeClassifier(max_depth=2, random_state=1), DecisionTreeClassifier(max_depth=3, random_state=1),
    ]
}

#Calling RandomizedSearchCV
randomized_cv = RandomizedSearchCV(estimator=Model, param_distributions=param_grid, n_iter=50, n_jobs = -1, scoring=scorer, cv=5, random_state=1)

#Fitting parameters in RandomizedSearchCV
randomized_cv.fit(X_train,y_train) 

print("Best parameters are {} with CV score={}:" .format(randomized_cv.best_params_,randomized_cv.best_score_))


# ### Tuning Adaboost: Oversampled

# In[52]:


# defining model
Model = AdaBoostClassifier(random_state=1)

# Parameter grid to pass in RandomSearchCV
param_grid = {
    "n_estimators": [100, 150, 200],
    "learning_rate": [0.2, 0.05],
    "base_estimator": [DecisionTreeClassifier(max_depth=1, random_state=1), DecisionTreeClassifier(max_depth=2, random_state=1), DecisionTreeClassifier(max_depth=3, random_state=1),
    ]
}

#Calling RandomizedSearchCV
randomized_cv = RandomizedSearchCV(estimator=Model, param_distributions=param_grid, n_iter=50, n_jobs = -1, scoring=scorer, cv=5, random_state=1)

#Fitting parameters in RandomizedSearchCV
randomized_cv.fit(X_train_over,y_train_over) 

print("Best parameters are {} with CV score={}:" .format(randomized_cv.best_params_,randomized_cv.best_score_))


# ### Tuning Adaboost: Undersampled

# In[53]:


# defining model
Model = AdaBoostClassifier(random_state=1)

# Parameter grid to pass in RandomSearchCV
param_grid = {
    "n_estimators": [100, 150, 200],
    "learning_rate": [0.2, 0.05],
    "base_estimator": [DecisionTreeClassifier(max_depth=1, random_state=1), DecisionTreeClassifier(max_depth=2, random_state=1), DecisionTreeClassifier(max_depth=3, random_state=1),
    ]
}

#Calling RandomizedSearchCV
randomized_cv = RandomizedSearchCV(estimator=Model, param_distributions=param_grid, n_iter=50, n_jobs = -1, scoring=scorer, cv=5, random_state=1)

#Fitting parameters in RandomizedSearchCV
randomized_cv.fit(X_train_un,y_train_un) 

print("Best parameters are {} with CV score={}:" .format(randomized_cv.best_params_,randomized_cv.best_score_))


# ### Tuning Logistic Regression: Original

# In[54]:


# defining model
Model = LogisticRegression(random_state=1)

#Parameter grid to pass in RandomSearchCV
param_grid = {'C': np.arange(0.1,1.1,0.1)}

#Calling RandomizedSearchCV
randomized_cv = RandomizedSearchCV(estimator=Model, param_distributions=param_grid, scoring=scorer, n_iter=50, n_jobs = -1, cv=5, random_state=1)

#Fitting parameters in RandomizedSearchCV
randomized_cv.fit(X_train, y_train)

print("Best parameters are {} with CV score={}:" .format(randomized_cv.best_params_,randomized_cv.best_score_))


# ### Tuning Logistic Regression: Oversampled

# In[55]:


# defining model
Model = LogisticRegression(random_state=1)

#Parameter grid to pass in RandomSearchCV
param_grid = {'C': np.arange(0.1,1.1,0.1)}

#Calling RandomizedSearchCV
randomized_cv = RandomizedSearchCV(estimator=Model, param_distributions=param_grid, scoring=scorer, n_iter=50, n_jobs = -1, cv=5, random_state=1)

#Fitting parameters in RandomizedSearchCV
randomized_cv.fit(X_train_over, y_train_over)

print("Best parameters are {} with CV score={}:" .format(randomized_cv.best_params_,randomized_cv.best_score_))


# ### Tuning Logistic Regression: Undersampled 

# In[56]:


# defining model
Model = LogisticRegression(random_state=1)

#Parameter grid to pass in RandomSearchCV
param_grid = {'C': np.arange(0.1,1.1,0.1)}

#Calling RandomizedSearchCV
randomized_cv = RandomizedSearchCV(estimator=Model, param_distributions=param_grid, scoring=scorer, n_iter=50, n_jobs = -1, cv=5, random_state=1)

#Fitting parameters in RandomizedSearchCV
randomized_cv.fit(X_train_un, y_train_un)

print("Best parameters are {} with CV score={}:" .format(randomized_cv.best_params_,randomized_cv.best_score_))


# ## Model Performance comparison
# 
# _**Choose best score of each classifier for a total of 6 chosen models.**_
# 
# #### Bagging: _**Oversampled data CV score=0.983583873824214:**_
# - 0.7323452555936634 with original data
# - 0.8885758615057977 with undersampled data
# 
# 
# #### Decision Tree: _**Oversampled data CV score=0.8740479375486185:**_
# - 0.4956230605912134 with original data
# - 0.809186673199412 with undersampled data
# 
# #### Random Forest:  _**Oversampled data CV score=0.9823736436211773**_
# - 0.6939408786542545 with original data
# - 0.8924220153519518 with undersampled data
# 
# #### Gradient Boosting:  _**Oversampled data CV score=0.9693622808629307:**_
# - 0.7593009962436714 with original data
# - 0.8975420545484241 with undersampled data
# 
# #### Adaboost:  _**Oversampled data CV score=0.9740525167670946:**_
# - 0.7708149599869344 with original data
# - 0.8885676955740649 with undersampled data
# 
# #### Logistic Regression:  _**Oversampled data CV score=0.87941624122865:**_
# - 0.5044667646578475 with original data
# - 0.8488567695574065 with undersampled data
# 
# #### Bagging

# In[62]:


# BAGGING: Create new pipeline with best parameters
bg_tuned = BaggingClassifier(
    n_estimators=70,
    max_samples=0.8,
    max_features=0.8)

bg_tuned.fit(X_train_over, y_train_over)


# In[71]:


# Check performance on oversampled train set
bg_train_perf = model_performance_classification_sklearn(bg_tuned, X_train_over, y_train_over)
bg_train_perf


# In[72]:


# Check performance on validation set
bg_val_perf = model_performance_classification_sklearn(bg_tuned, X_val, y_val)
bg_val_perf


# ### Decision Tree

# In[65]:


# DECISION TREE: Create new pipeline with best parameters
dtree_tuned = DecisionTreeClassifier(
    min_samples_leaf=7,
    min_impurity_decrease=0.001,
    max_leaf_nodes=15,
    max_depth=3)

dtree_tuned.fit(X_train_over, y_train_over)


# In[73]:


# Check performance on oversampled train set
dtree_train_perf = model_performance_classification_sklearn(dtree_tuned, X_train_over, y_train_over)
dtree_train_perf


# In[74]:


# Check performance on validation set
dtree_val_perf = model_performance_classification_sklearn(dtree_tuned, X_val, y_val)
dtree_val_perf


# #### Random Forest

# In[60]:


# RANDOM FOREST: Create new pipeline with best parameters
rf_tuned = RandomForestClassifier(
    max_features=0.3,
    random_state=1,
    max_samples=0.6,
    n_estimators=300,
    min_samples_leaf=1,)

rf_tuned.fit(X_train_over, y_train_over) 


# In[75]:


# Check performance on oversampled train set
rf_train_perf = model_performance_classification_sklearn(rf_tuned, X_train_over, y_train_over)
rf_train_perf


# In[76]:


# Check performance on validation set
rf_val_perf = model_performance_classification_sklearn(rf_tuned, X_val, y_val)
rf_val_perf


# #### Gradient Boosting

# In[67]:


# GRADIENT BOOSTING: Create new pipeline with best parameters
gbm_tuned = GradientBoostingClassifier(
    subsample=0.7, 
    n_estimators=125,
    max_features=0.5,
    learning_rate=1)

gbm_tuned.fit(X_train_over, y_train_over)


# In[77]:


# Check performance on oversampled train set
gbm_train_perf = model_performance_classification_sklearn(gbm_tuned, X_train_over, y_train_over)
gbm_train_perf


# In[78]:


# Check performance on validation set
gbm_val_perf = model_performance_classification_sklearn(gbm_tuned, X_val, y_val)
gbm_val_perf


# #### Adaboost

# In[68]:


# ADABOOST: Create new pipeline with best parameters
ada_tuned = AdaBoostClassifier(
    n_estimators= 200, 
    learning_rate= 0.2, 
    base_estimator= DecisionTreeClassifier(max_depth=3,random_state=1)
) 

ada_tuned.fit(X_train_over, y_train_over) 


# In[79]:


# Check performance on oversampled train set
ada_train_perf = model_performance_classification_sklearn(ada_tuned, X_train_over, y_train_over)
ada_train_perf


# In[80]:


# Check performance on validation set
ada_val_perf = model_performance_classification_sklearn(ada_tuned, X_val, y_val)
ada_val_perf


# #### Logistic Regression

# In[69]:


# LOGISTIC REGRESSION: Create new pipeline with best parameters

log_tuned = LogisticRegression(C = 0.1)

log_tuned.fit(X_train_over, y_train_over)


# In[81]:


# Check performance on oversampled train set
log_train_perf = model_performance_classification_sklearn(log_tuned, X_train_over, y_train_over)
log_train_perf


# In[82]:


# Check performance on validation set
log_val_perf = model_performance_classification_sklearn(log_tuned, X_val, y_val)
log_val_perf


# In[87]:


# training performance comparison

models_train_comp_df = pd.concat(
    [
        bg_train_perf.T,
        dtree_train_perf.T,
        rf_train_perf.T,
        gbm_train_perf.T,
        ada_train_perf.T, 
        log_train_perf.T
    ],
    axis=1,
)

models_train_comp_df.columns = [
    "Bagging tuned with oversampled data",
    "Decision Tree tuned with oversampled data",
    "Random forest tuned with oversampled data",
    "Gradient Boosting tuned with oversampled data",
    "AdaBoost classifier tuned with oversampled data",
    "Logistic Regression tuned with oversampled data",
]
print("Training performance comparison:")
models_train_comp_df


# In[91]:


# validation performance comparison

models_val_comp_df = pd.concat(
    [
        bg_val_perf.T,
        dtree_val_perf.T,
        rf_val_perf.T,
        gbm_val_perf.T,
        ada_val_perf.T, 
        log_val_perf.T
    ],
    axis=1,
)

models_val_comp_df.columns = [
    "Bagging tuned with oversampled data",
    "Decision Tree tuned with oversampled data",
    "Random forest tuned with oversampled data",
    "Gradient Boosting tuned with oversampled data",
    "AdaBoost classifier tuned with oversampled data",
    "Logistic Regression tuned with oversampled data",
]
print("Validation performance comparison:")
models_val_comp_df


# In[107]:


### Important features of the final model

feature_names = X_train.columns
importances = ada_tuned.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(12,12))
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='violet', align='center')
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()


# ## Pipelines to build the final model

# In[109]:


# Create pipeline for the best model

Model = Pipeline([
    ('scaler',StandardScaler()),
    ('clf', AdaBoostClassifier())
])


# In[110]:


X1 = df.drop(columns="Target")
Y1 = df["Target"]

# Built an existing test set above, don't need to divide data here 

X_test1 = test.drop(columns="Target")
y_test1 = test["Target"]


# In[111]:


# impute missing values in X1

X1 = imputer.fit_transform(X1)


# In[112]:


sm = SMOTE(sampling_strategy=1, k_neighbors=5, random_state=1)
X_over1, y_over1 = sm.fit_resample(X1, Y1)


# In[113]:


Model.fit(X_train,y_train)


# In[114]:


# pipeline object's accuracy on the train set

Model.score(X_train, y_train)


# In[115]:


# pipeline object's accuracy on the test set

Model.score(X_test, y_test)


# # Business Insights and Conclusions
# 
# Best model: Adaboost Classifier with oversampled data, highest recall score.
# - Recall: 87.2%
# - Accuracy: 98.4%
# - Precision: 83.7%
# - F1: 85.4%
# 
# The most important features noted in our chosen model are V36, V30, V18, V12, V9, and V35.
# 
# With high recall, we will be able to minimize false negatives, i.e. predicting a machine will have no failure when there will be a failure. This will control maintenance costs. 
# 
# Considering the predictor variables, ReneWind may point out important sensors to implement generator failure warning signs.
