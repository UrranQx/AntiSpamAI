# models/random_forest.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


class RandomForestSpamClassifier:
    def __init__(self, n_estimators=100, max_depth=None):
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )

    def train(self, X_train, y_train):
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        self.model.fit(X_train_tfidf, y_train)

    def predict(self, X_test):
        X_test_tfidf = self.vectorizer.transform(X_test)
        return self.model.predict(X_test_tfidf)

    def predict_proba(self, X_test):
        X_test_tfidf = self.vectorizer.transform(X_test)
        return self.model.predict_proba(X_test_tfidf)
