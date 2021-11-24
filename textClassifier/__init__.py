from textClassifier.automltext import text_model
from textClassifier.statistical_features_pipeline import FeatureMultiplierCount, stats_models
from textClassifier.scoring_methods import scorer_f1, \
scorer_accuracy, scorer_precision, scorer_recall, \
scorer_roc
from textClassifier.tfidf_features import tfidf_models
from textClassifier.clean_data import Cleaner

from textClassifier.all_features import FeatureMultiplierCount, all_features_models