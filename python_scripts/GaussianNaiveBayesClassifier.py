# -*- coding: utf-8 -*-

# TODO:
# - Implement partial_fit() function: the function should work for either one 
#   new sample or many. Make sure the resulting parameters are the same as 
#   using the standard fit() function.
# - Implement decision_boundary() function: this function should allow to 
#   easily visualize the decision boundary of the trained classifier
# - Normalize probability in predict_proba() so that the array sums to 1

import numpy as np

class GaussianNaiveBayesClassifier():
    """Gaussian Naive Bayes Classifier
    
    HEAVILY based on scikit-learn's implementation: 
    http://scikit-learn.org/stabl/modules/generated/sklearn.naive_bayes.GaussianNB.html#sklearn.naive_bayes.GaussianNB    
    
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
        """Fit the Gaussian Naive Bayes model
        
        Args:
            X (np.array) :
            y (np.array) :
        
        Note: Calling `fit()` overwrites previous information. Use `partial_fit()`
            to update the model with new training data.
        """
        
        # Compute class prior
        self.class_count_ = np.array([np.sum(y==label) for label in set(y)])
        self.class_prior_ = self.class_count_/float(self.class_count_.sum())
        
        # Compute the mean and variance of each class
        self.theta_ = np.array([np.mean(X[y==label,:], axis=0) for label in set(y)])
        self.sigma_ = np.array([np.var(X[y==label,:], axis=0) for label in set(y)])
        self._sse = self.sigma_*(self.class_count_ - 1).reshape(-1,1) # Sum of squared differences
    
    def partial_fit(self, X, y):
        """
        """
        
        if self.class_count_ is None: # model has not been trained yet, use `fit()`
            self.fit(X, y)
            
        else:    
            # Update class prior
            self.class_count_ += np.array([np.sum(y==label) for label in set(y)])
            self.class_prior_ = self.class_count_/float(self.class_count_.sum())
            
            # Update mean
            e = X - self.theta_
            self.theta_ += np.sum(e/self.class_count_)
            
            # Update variance
            self._sse += e*(X - self.theta_)
            self.sigma_ = self._sse/(self.class_count_ - 1)        
    
    def predict(self, X):
        return np.argmax(self.predict_proba(X))
    
    def predict_proba(self, X):
        return np.prod(self._gaussian(X, self.theta_, self.sigma_), axis=1)
    
    def get_params(self):
        return self.class_prior_, self.theta_, self.sigma_
    
    def set_params(self, class_prior=None, theta=None, sigma=None):
        
        if class_prior is not None:
            self.class_prior_ = class_prior

        if theta is not None:  
            self.theta_ = theta
            
        if sigma is not None:
            self.sigma_ = sigma
    
    def score(self, X, y):
        y_hat = self.predict(X)
        return np.mean(y == y_hat) # Accuracy
    
    def _gaussian(self, X, mu, var):
        """Probability of X ~ N(mu,var)
        
        Args:
            X (np.array) : values for which to compute the probability
            mu : mean of the Gaussian distribution
            var : variance of the Gaussian distribution
            
        Returns
            np.array
        """
        
        return np.exp(-(X-mu)**2/(2*var))/(np.sqrt(2*np.pi*var))
        
    def decision_boundary(self):
        """Define the decision boundary of the trained classifier.
        
        It should be a piecewise quadratic boundary
        """
        pass
        
    
if __name__ == '__main__':
    
    # 1. Create fake dataset
    nb_points = 200
    nb_features = 3
    
    X1 = 2*np.random.randn(nb_points/2, nb_features) + 1
    y1 = np.zeros((X1.shape[0],))
    
    X2 = 3*np.random.randn(nb_points/2, nb_features) + 3
    y2 = np.ones((X2.shape[0],))
    
    X = np.concatenate((X1, X2), axis=0)
    y = np.concatenate((y1, y2), axis=0)

    # Initialize and train classifier
    clf = GaussianNaiveBayesClassifier()
    clf.fit(X, y)
    
    # Test classifier
    X_test = np.random.rand(1, nb_features) + 2
    
    print(X_test)
    print(clf.predict_proba(X_test))
    print(clf.predict(X_test))
    