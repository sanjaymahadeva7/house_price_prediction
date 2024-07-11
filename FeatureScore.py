import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

class FeatureScoreCalculator:
    def __init__(self, data, target, feature_names=None):
        self.data = data.copy() 
        self.target = target
        self.feature_names = feature_names if feature_names else data.columns.drop(target)
        self._preprocess_data()
        self.scores = self._calculate_correlation_scores()

    def _preprocess_data(self):
        # Convert all non-numeric features to numeric using label encoding
        label_encoders = {}
        for column in self.feature_names:
            if self.data[column].dtype == 'object':
                label_encoders[column] = LabelEncoder()
                self.data[column] = label_encoders[column].fit_transform(self.data[column].astype(str))
        
        # Convert all features to numeric, invalid parsing will be set as NaN
        self.data[self.feature_names] = self.data[self.feature_names].apply(pd.to_numeric, errors='coerce')
        
        # Fill NaN values with the mean of the respective column
        self.data[self.feature_names] = self.data[self.feature_names].fillna(self.data[self.feature_names].mean())
        
        # Ensure target is binary encoded
        if self.data[self.target].dtype == 'object':
            target_encoder = LabelEncoder()
            self.data[self.target] = target_encoder.fit_transform(self.data[self.target].astype(str))

        # Standardize the features
        scaler = StandardScaler()
        self.data[self.feature_names] = scaler.fit_transform(self.data[self.feature_names])
        
    def _calculate_correlation_scores(self):
        features = self.feature_names
        target = self.target
        data = self.data
        scores = []

        for feature in features:
            correlation_score = np.abs(data[feature].corr(data[target]))
            scores.append(correlation_score)
        
        return scores

    def display_scores(self):
        if len(self.scores) != len(self.feature_names):
            raise ValueError("The number of scores does not match the number of features")

   
        max_score = max(self.scores)
        scaled_scores = [score / max_score * 100 for score in self.scores]

        feature_scores = list(zip(self.feature_names, scaled_scores))
        feature_scores_sorted = sorted(feature_scores, key=lambda x: x[1], reverse=True)

        for i, (name, score) in enumerate(feature_scores_sorted, start=1):
            print(f"Feature {i}: {name} - Correlation Score = {score:.2f}")
