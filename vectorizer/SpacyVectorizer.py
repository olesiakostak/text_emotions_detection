import numpy as np
import spacy
from sklearn.base import BaseEstimator, TransformerMixin

nlp = spacy.load("en_core_web_md")

class SpacyVectorizer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([nlp(text).vector for text in X])