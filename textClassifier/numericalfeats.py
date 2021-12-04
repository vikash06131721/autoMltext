from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import joblib
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import mean_squared_error, make_scorer, f1_score
import pickle
from sklearn.pipeline import Pipeline
# from sklearn.externals import joblib
from math import sqrt

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings("ignore")


#modelling pipeline 
def models_numerical(scoring_method,cv):
    """
    
    returns three pipelines Random Forest, Adaboost, Gradientboost,SVC
    The scoring function is based on f1_score
    """


    # Random Forest Pipeline
    rf_pipeline = Pipeline([
    ('u1', FeatureUnion([
 
        ('numerical_features',Pipeline([
                                       ('scaler',StandardScaler()),
                                       ])),

    ])),
    ('clf', RandomForestClassifier()),])
    # Adaboost Pipeline
    AdaBoost_pipeline = Pipeline([
    ('u1', FeatureUnion([
 
        ('numerical_features',Pipeline([
                                       ('scaler',StandardScaler()),
                                       ])),

    ])),
    ('clf', AdaBoostClassifier()),])
    # Gradient Boost Pipeline
    GRD_pipeline = Pipeline([
    ('u1', FeatureUnion([
 
        ('numerical_features',Pipeline([
                                       ('scaler',StandardScaler()),
                                       ])),

    ])),
    ('clf', GradientBoostingClassifier()),])
    
    svm_pipeline =Pipeline([
    ('u1', FeatureUnion([
 
        ('numerical_features',Pipeline([
                                       ('scaler',StandardScaler()),
                                       ])),

    ])),
    ('clf', LinearSVC()),])

    logistic_pipeline = Pipeline([
    ('u1', FeatureUnion([
 
        ('numerical_features',Pipeline([
                                       ('scaler',StandardScaler()),
                                       ])),

    ])),
    ('clf', LogisticRegression()),])

    dt_pipeline = Pipeline([
    ('u1', FeatureUnion([
 
        ('numerical_features',Pipeline([
                                       ('scaler',StandardScaler()),
                                       ])),

    ])),
    ('clf', DecisionTreeClassifier()),])

    #grid search params for randomforest, adaboost, gradientboost
    # import pdb;pdb.set_trace()
    grid_params_rf = [{'clf__n_estimators': [10, 50, 100], 'clf__max_depth': [2, 3, 5]}]
    grid_params_adaboost = [{'clf__n_estimators': [10, 50, 100,500], 'clf__learning_rate': [0.5, 0.8, 1.0]}]
    grid_params_grd = [{'clf__n_estimators': [10, 50, 100,500], 'clf__learning_rate': [0.5, 0.8, 1.0],
                            'clf__max_depth': [2, 3, 5]}]
    
    grid_params_svc = [{'clf__C': [1.0,3.0,5.0,10.0],'clf__max_iter':[100,1000,10000]}]

    grid_params_logistic =[{'clf__C': [1.0,3.0,5.0,10.0],'clf__max_iter':[100,1000,10000]}]

    grid_params_dt =[{'clf__max_depth': [2,3,5],'clf__max_features':["auto","sqrt","log2"]}]


    #gridsearchcv pipeline for randomforest
    gs_rf = GridSearchCV(estimator=rf_pipeline,
                             param_grid=grid_params_rf,
                             scoring=scoring_method,
                             cv=cv)
    #gridsearchcv pipeline for adaboost
    gs_adaboost = GridSearchCV(estimator=AdaBoost_pipeline,
                                   param_grid=grid_params_adaboost,
                                   scoring=scoring_method,
                                   cv=cv)

    #gridsearchcv pipeline for gradientboost
    gs_grd = GridSearchCV(estimator=GRD_pipeline,
                              param_grid=grid_params_grd,
                              scoring=scoring_method,
                              cv=cv)
    gs_svc = GridSearchCV(estimator=svm_pipeline,
                              param_grid=grid_params_svc,
                              scoring=scoring_method,
                              cv=cv)

    gs_logistic = GridSearchCV(estimator=logistic_pipeline,
                              param_grid=grid_params_logistic,
                              scoring=scoring_method,
                              cv=cv)
    gs_decision = GridSearchCV(estimator=dt_pipeline,
                              param_grid=grid_params_dt,
                              scoring=scoring_method,
                              cv=cv)

    grids = [gs_svc, gs_adaboost, gs_rf,gs_grd,gs_logistic,gs_decision]
    # grids = [gs_rf]
    return grids