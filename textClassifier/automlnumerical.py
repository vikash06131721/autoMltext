import numpy as np
import pandas as pd
from textClassifier.numericalfeats import models_numerical
from functools import reduce
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score,\
recall_score, roc_auc_score, accuracy_score


class numerical_model():
    
    def __init__(self,numerical_columns,
                 categorical_columns,
                 index_column,
                 target_column,
                 data_path,
                 scoring,
                 cv):
        self.numerical_columns = numerical_columns
        self.categorical_columns = categorical_columns
        self.index_column = index_column
        self.target_column = target_column
        self.data_path = data_path
        self.scoring =scoring
        self.cv = cv
        
    def read_dataframe(self):
        return pd.read_csv(self.data_path)

    def __select_scoring(self):
        scoring_fn_dicts ={"f1":"f1",
        "acc":"accuracy",
        "precision":"precision",
        "recall":"recall",
        "roc":"roc_auc"}

        return scoring_fn_dicts
        
    def __separate_categorical_numerical_dataframes(self):
        
        dataframe = self.read_dataframe()
        
        categorical_columns = self.categorical_columns + self.index_column
        numerical_columns = self.numerical_columns + self.index_column
        index_column = self.index_column + self.target_column

        categorical_df = dataframe[categorical_columns]
        numerical_df = dataframe[numerical_columns]
        target_df = dataframe[index_column]

        return categorical_df, numerical_df, target_df
    
    def fill_na_median(self,numerical_df):
        columns_consider =[]
        for k,v in numerical_df.isnull().sum().items():
            if v>0:
                columns_consider.append(k)

        for cols in columns_consider:
            med_value = np.nanmedian(numerical_df[cols].values)
            numerical_df[cols] = numerical_df[cols].fillna(med_value)
        return numerical_df
    
    def encode_categorical_columns(self):
        categorical_df = self.__separate_categorical_numerical_dataframes()[0]
        numerical_df = self.__separate_categorical_numerical_dataframes()[1]
        target_df = self.__separate_categorical_numerical_dataframes()[2]
        
        encoded_df = pd.get_dummies(categorical_df,
                                   columns = self.categorical_columns)
        numerical_df = self.fill_na_median(numerical_df)
        return encoded_df, numerical_df, target_df

    def build_model(self,test_size,model_save_path):

        #readin data
        categorical = self.encode_categorical_columns()[0]
        numerical = self.encode_categorical_columns()[1]
        target = self.encode_categorical_columns()[2]

        data_frames = [categorical, numerical, target]

        df_merged = reduce(lambda  left,right: pd.merge(left,right,on=self.index_column,
                                            how='inner'), data_frames)

        df_merged.set_index(self.index_column,inplace=True)
        target = df_merged.pop(self.target_column[0])
        
        X_train, X_test,y_train, y_test =\
        train_test_split(df_merged, target, test_size=test_size,random_state=42)

        best_acc = 0.0
        best_clf = 0
        best_gs = ''
        grid_dict = {0: 'svc',1:'adaBoost',2:'randomForest',3:'gradientBoost',4:'logisticRegression',
        5:'decisionTree'}

        scoring_method = self.__select_scoring()[self.scoring]
        grids = models_numerical(scoring_method=scoring_method,\
        cv=self.cv)
        
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
                best_acc = score
            
                best_gs = gs
                best_clf = idx
        print('\Classifier with best test set Acc score: %s' % grid_dict[best_clf])
        joblib.dump(best_gs,model_save_path)

