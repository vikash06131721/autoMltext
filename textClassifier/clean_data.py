import re
import pandas as pd
import stop_words
import re
import nltk
from nltk.stem import PorterStemmer 
stopwords =stop_words.get_stop_words(language='en')
wpt = nltk.WordPunctTokenizer()
ps = PorterStemmer() 
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from sklearn.base import BaseEstimator, TransformerMixin


class Cleaner(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def clean_and_normalize_text_data(self,sent):
            
        sent_cleaned= re.sub(r'http\S+', '', " ".join(sent))

        sent_cleaned = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", " ", sent)
        sent_cleaned = sent_cleaned.replace('\n','')
        sent_cleaned = sent_cleaned.strip(' ').lower()

        stop_remove=[]
        for word in sent_cleaned.split(' '):
            if word.strip(' ') not in stopwords:
                stop_remove.append(word)

        return ' '.join(stop_remove)

    def transform(self, X,y=None):
        c=pd.Series([self.clean_and_normalize_text_data(x) for x in X])
        

        return c

    def fit(self, *_):
        return self