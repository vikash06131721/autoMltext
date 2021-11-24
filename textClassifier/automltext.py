import pandas as pd
import joblib
import numpy as np
import ast
import nltk
import re
import argparse
from textClassifier.tfidf_features import tfidf_models
from textClassifier.all_features import all_features_models
from textClassifier.statistical_features_pipeline import \
FeatureMultiplierCount, stats_models

from textClassifier.scoring_methods import scorer_f1, scorer_accuracy, scorer_precision, scorer_recall, scorer_roc
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import mean_squared_error, make_scorer, f1_score
import pickle
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score
# from sklearn.externals import joblib
from math import sqrt
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_validate
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import StandardScaler

import random
import stop_words
from nltk.stem import PorterStemmer 
stops =stop_words.get_stop_words(language='en')
wpt = nltk.WordPunctTokenizer()
ps = PorterStemmer() 
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
import warnings
warnings.filterwarnings("ignore") 


class text_model():

    def __init__(self,data_path,scoring,features,cv):
        self.data_path = data_path
        self.scoring =scoring
        self.features =features
        self.cv = cv
        
    def __select_scoring(self):
        scoring_fn_dicts ={"f1":"f1",
        "acc":"accuracy",
        "precision":"precision",
        "recall":"recall",
        "roc":"roc_auc"}

        return scoring_fn_dicts

    def __select_features(self):
        if self.features=="statistical":
            model = stats_models(scoring_method= \
            self.__select_scoring()[self.scoring],cv=self.cv)
        elif self.features=="tfidf":
            model = tfidf_models(scoring_method= \
            self.__select_scoring()[self.scoring],cv =self.cv)
        elif self.features=="all":
            model = all_features_models(scoring_method= \
            self.__select_scoring()[self.scoring],cv = self.cv)
        else:
            model =None
        return model

    def build_model(self, text_column, target_column,test_size):
        print ("Internally we rename the text column to 'sentences' and target column to 'target'")
        train_data = pd.read_csv(self.data_path)
        train_data.rename(columns={text_column:"sentences",\
        target_column:"target"},inplace=True)

        X=train_data.sentences
        y=train_data.target
        
        X_train, X_test,y_train, y_test = train_test_split(X, y, test_size=test_size,random_state=42)

        best_acc = 0.0
        best_clf = 0
        best_gs = ''
        grid_dict = {0: 'svc',1:'adaBoost',2:'randomForest',3:'gradientBoost',4:'logisticRegression',
        5:'decisionTree'}

        grids = self.__select_features()
        
        for idx, gs in enumerate(grids):
            print('\nEstimator: %s' % grid_dict[idx])
            gs.fit(X_train, y_train)
            print('Best params: %s' % gs.best_params_)

            print('Best training  score: %.3f' % gs.best_score_)
            y_pred = gs.predict(X_test)
            if self.scoring=='f1':
                score = f1_score(y_test, y_pred)
                print('Test set f1 score for best params: %.3f ' % score)
            elif self.scoring=='acc':
                score = accuracy_score(y_test, y_pred)
                print('Test set acc score for best params: %.3f ' % score)
            elif self.scoring=='precision':
                score = precision_score(y_test, y_pred)
                print('Test set precision score for best params: %.3f ' % score)
            elif self.scoring=='recall':
                score = recall_score(y_test, y_pred)
                print('Test set recall score for best params: %.3f ' % score)
            elif self.scoring=='roc':
                score = roc_auc_score(y_test, y_pred)
                print('Test set roc score for best params: %.3f ' % score)
            else:
                print ("no valid scoring method")
                break
            if score >best_acc:
                best_acc = f1_score(y_test, y_pred)
            
                best_gs = gs
                best_clf = idx
        print('\Classifier with best test set Acc score: %s' % grid_dict[best_clf])

            



