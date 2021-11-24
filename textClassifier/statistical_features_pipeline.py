from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import joblib
import numpy as np
import ast
import nltk
import re
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

class FeatureMultiplierCount(BaseEstimator, TransformerMixin):
    def __init__(self, word_count=True,char_count=True,
                word_density=True,total_length=True,
                capitals=True,caps_vs_length=True,num_exclamation_marks=True,num_question_marks=True,
                num_punctuation=True,num_symbols=True,num_unique_words=True,words_vs_unique=True,
                word_unique_percent=True):
        self.word_count = word_count
        self.total_length = total_length
        self.char_count =char_count
        self.word_density = word_density
        self.capitals = capitals
        self.caps_vs_length = caps_vs_length
        self.num_exclamation_marks=num_exclamation_marks
        self.num_question_marks=num_question_marks
        self.num_punctuation=num_punctuation
        self.num_symbols=num_symbols
        self.num_unique_words = num_unique_words
        self.words_vs_unique = words_vs_unique
        self.word_unique_percent = word_unique_percent

    def transform(self, X,y=None):
        X = pd.DataFrame(X)
        X['word_count'] = X['sentences'].apply(lambda x : len(x.split()))
        X['char_count'] = X['sentences'].apply(lambda x : len(x.replace(" ","")))
        X['word_density'] = X['word_count'] / (X['char_count'] + 1)

        X['total_length'] = X['sentences'].apply(len)
        X['capitals'] = X['sentences'].apply(lambda comment: sum(1 for c in comment if c.isupper()))
 
        X['num_exclamation_marks'] =X['sentences'].apply(lambda x: x.count('!'))
        X['num_question_marks'] = X['sentences'].apply(lambda x: x.count('?'))
        X['num_punctuation'] = X['sentences'].apply(lambda x: sum(x.count(w) for w in '.,;:'))
        X['num_symbols'] = X['sentences'].apply(lambda x: sum(x.count(w) for w in '*&$%'))
        X['num_unique_words'] = X['sentences'].apply(lambda x: len(set(w for w in x.split())))
       
        X.fillna(0,inplace=True)

        return X[['word_count','char_count','word_density','total_length',\
        'capitals','num_exclamation_marks','num_question_marks','num_punctuation',\
        'num_symbols',
                 'num_unique_words']]

    def fit(self, *_):
        return self

#modelling pipeline 
def stats_models(scoring_method,cv):
    """
    
    returns three pipelines Random Forest, Adaboost, Gradientboost,SVC
    The scoring function is based on f1_score
    """


    # Random Forest Pipeline
    rf_pipeline = Pipeline([
    ('u1', FeatureUnion([
        # ('tfdif_features', Pipeline([
        #      ('tfidf', TfidfVectorizer(max_features=10000,ngram_range=(2,5))),
        # ])),
        ('numerical_features',Pipeline([('numerical_feats',FeatureMultiplierCount()),
                                       ('scaler',StandardScaler()),
                                       ])),

    ])),
    ('clf', RandomForestClassifier()),])
    # Adaboost Pipeline
    AdaBoost_pipeline = Pipeline([
    ('u1', FeatureUnion([
        # ('tfdif_features', Pipeline([
        #      ('tfidf', TfidfVectorizer(max_features=10000,ngram_range=(2,5))),
        # ])),
        ('numerical_features',Pipeline([('numerical_feats',FeatureMultiplierCount()),
                                       ('scaler',StandardScaler()),
                                       ])),

    ])),
    ('clf', AdaBoostClassifier()),

])
    # Gradient Boost Pipeline
    GRD_pipeline = Pipeline([
    ('u1', FeatureUnion([
        # ('tfdif_features', Pipeline([
        #      ('tfidf', TfidfVectorizer(max_features=10000,ngram_range=(2,5))),
        # ])),
        ('numerical_features',Pipeline([('numerical_feats',FeatureMultiplierCount()),
                                       ('scaler',StandardScaler()),
                                       ])),

    ])),
    ('clf', GradientBoostingClassifier()),

])
    
    svm_pipeline =Pipeline([
    ('u1', FeatureUnion([
        # ('tfdif_features', Pipeline([
        #      ('tfidf', TfidfVectorizer(max_features=10000,ngram_range=(1,3))),
        # ])),
        ('numerical_features',Pipeline([('numerical_feats',FeatureMultiplierCount()),
                                       ('scaler',StandardScaler()),
                                       ])),

    ])),
    ('clf', LinearSVC(max_iter=10000)),])

    logistic_pipeline = Pipeline([
    ('u1', FeatureUnion([
        # ('tfdif_features', Pipeline([
        #      ('tfidf', TfidfVectorizer(max_features=10000,ngram_range=(1,3))),
        # ])),
        ('numerical_features',Pipeline([('numerical_feats',FeatureMultiplierCount()),
                                       ('scaler',StandardScaler()),
                                       ])),

    ])),
    ('clf', LogisticRegression()),])

    dt_pipeline = Pipeline([
    ('u1', FeatureUnion([
        # ('tfdif_features', Pipeline([
        #      ('tfidf', TfidfVectorizer(max_features=10000,ngram_range=(1,3))),
        # ])),
        ('numerical_features',Pipeline([('numerical_feats',FeatureMultiplierCount()),
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
