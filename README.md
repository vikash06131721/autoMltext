# Automltext


textClassifier is a package used to generate base result for text data for binary classification

- from textClassifier.automltext import text_model
- automl = text_model(data_path=data_path,scoring='f1',features='all',cv=5,model_save_path="model.joblib")
- automl.build_model(text_column='text',target_column='airline_sentiment',test_size=0.20)

Configurable arguments:

1. data_path : csv data path
2. scoring : ["f1","acc","precision","recall","roc"]
3. features: ["all","statistical","tfidf"]
4. cv = [2,3,5,7,10]

# Features:
The features here are statistical i.e the below features are calculated:
- 'word_count','char_count','word_density','total_length','capitals','num_exclamation_marks','num_question_marks','num_punctuation','num_symbols','num_unique_words'

We also calculate tfidf based features.

The option is there to select either of them("statistical", "tfidf") or concatenate all the features("all")

# AutomlNumerical

- from textClassifier.automlnumerical import numerical_model
- num_model = numerical_model(numerical_columns=numerical_columns,
                           categorical_columns=categorical_columns,
                           index_column=index_column,
                            target_column=target_column,
                            data_path=" ",
                            scoring = "f1",
                            cv = 5
                           )

- num_model.build_model(test_size=0.25,model_save_path=" ")

1. numerical_columns = list of columns

2. categorical_columns = list of columns

3. index_column = list of column(only one column can be index like customer id)

4. target_column = list of column

5. data_path = data path

6. scoring = ["f1","acc","precision","recall","roc"]

7. cv = [2,3,5,7,10]

8. test_size = float point

9. model_save_path = model path to save


## Installation
1. git clone https://github.com/vikash06131721/autoMltext.git

2. cd autoMltext/
3. python setup.py install
## License
MIT



