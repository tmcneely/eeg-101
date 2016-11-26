# -*- coding: utf-8 -*-

import numpy as np

class GaussianNaiveBayesClassifier():
    """Gaussian Naive Bayes Classifier
    
    HEAVILY based on scikit-learn's implementation: 
    http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html#sklearn.naive_bayes.GaussianNB    
    
    Args:
        priors
        
    Attributes:
        class_prior_
        class_count_
        theta_
        sigma_
    
    """
    
    def __init__(self, priors=None):
        
        self.class_prior_ = []
        self.class_count_ = []
        self.theta_ = [] # Mean of each feature per class
        self.sigma_ = [] # Variance of each feature per class
    
    def fit(self, X, y):
        
        # Compute class prior
        self.class_count_ = np.array([np.sum(y==label) for label in set(y)])
        self.class_prior_ = self.class_count_/float(self.class_count_.sum())
        
        # Compute the mean and variance of each class
        self.theta_ = np.array([np.mean(X[y==label,:], axis=0) for label in set(y)])
        self.sigma_ = np.array([np.var(X[y==label,:], axis=0) for label in set(y)])
    
    def partial_fit(self, X, y):
        pass
    
    def predict(self, X):
        return np.argmax(self.predict_proba(X))
    
    def predict_proba(self, X):
        return np.prod(self._gaussian(X, self.theta_, self.sigma_), axis=1)
    
    def get_params(self):
        pass
    
    def set_params(self):
        pass
    
    def score(self):
        pass
    
    def _gaussian(self, X, mu, sigma):
        return np.exp(-(X-mu)**2/(2*sigma**2))/(np.sqrt(2*np.pi)*sigma)
        
    
if __name__ == '__main__':
    
    nb_points = 200
    nb_features = 10
    
    X = np.random.rand(nb_points, nb_features)
    y = np.random.binomial(1, 0.5, nb_points)
    
    clf = GaussianNaiveBayesClassifier()
    clf.fit(X, y)
    
    print(clf.predict_proba(np.random.rand(1, nb_features)))
    
    print(clf.predict(np.random.rand(1, nb_features)))
    