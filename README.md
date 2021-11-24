# Automltext


textClassifier is a package used to generate base result for text data for binary classification

- from textClassifier.automltext import text_model
- automl = text_model(data_path=data_path,scoring='f1',features='all',cv=5)
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

## Installation
1. git clone https://github.com/vikash06131721/autoMltext.git

2. cd autoMltext/
3. python setup.py install
## License
MIT



