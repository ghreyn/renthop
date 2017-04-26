#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 16:22:21 2017

@author: reynold
"""
import pandas as pd
import numpy as np
import random

#Our feature construction class will inherit from these two base classes of sklearn.
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin

SEED=2017

class manager_rating(BaseEstimator, TransformerMixin):
    """
    Adds the column "manager_skill" to the dataset, based on the Kaggle kernel
    "Improve Perfomances using Manager features" by den3b. The function should
    be usable in scikit-learn pipelines.
    
    Parameters
    ----------
    threshold : Minimum count of rental listings a manager must have in order
                to get his "own" score, otherwise the mean is assigned.

    Attributes
    ----------
    mapping : pandas dataframe
        contains the manager_skill per manager id.
        
    mean_skill : float
        The mean skill of managers with at least as many listings as the 
        threshold.
    """
    def __init__(self, threshold = 5):
        
        self.threshold = threshold
        
    def _reset(self):
        """Reset internal data-dependent state of the scaler, if necessary.
        
        __init__ parameters are not touched.
        """
        # Checking one attribute is enough, becase they are all set together
        # in fit        
        if hasattr(self, 'mapping_'):
            
            self.mapping_ = {}
            self.mean_skill_ = 0.0
        
    def fit(self, X,y):
        """Compute the skill values per manager for later use.
        
        Parameters
        ----------
        X : pandas dataframe, shape [n_samples, n_features]
            The rental data. It has to contain a column named "manager_id".
            
        y : pandas series or numpy array, shape [n_samples]
            The corresponding target values with encoding:
            low: 0.0
            medium: 1.0
            high: 2.0
        """        
        self._reset()
        
        temp = pd.concat([X.manager_id,pd.get_dummies(y)], axis = 1).groupby('manager_id').mean()
        temp['count'] = X.groupby('manager_id').size()
        
        temp['manager_skill'] = temp['high']*2 + temp['medium']
        
        mean = temp.loc[temp['count'] >= self.threshold, 'manager_skill'].mean()
#        print ("threshold: {} mean manager rating: {}".format(self.threshold, mean))
        
        temp.loc[temp['count'] < self.threshold, 'manager_skill'] = mean
        
        self.mapping_ = temp[['manager_skill']]
        self.mean_skill_ = mean
            
        return self
        
    def transform(self, X):
        """Add manager skill to a new matrix.
        
        Parameters
        ----------
        X : pandas dataframe, shape [n_samples, n_features]
            Input data, has to contain "manager_id".
        """        
        X = pd.merge(left = X, right = self.mapping_, how = 'left', left_on = 'manager_id', right_index = True)
        X['manager_skill'].fillna(self.mean_skill_, inplace = True)
        
        return X


def mgr_rating_pct(dftrain, dftest):
    """
    """

    index = list(range(dftrain.shape[0]))
    random.seed(SEED)
    random.shuffle(index)     
        
    dftrain['mgr_low_pct'] = np.nan
    dftrain['mgr_med_pct'] = np.nan
    dftrain['mgr_high_pct'] = np.nan
        
    # use train folds to compute values for train folds
    for i in range(5):
        # compute fold indices
        test_idx = index[int((i * dftrain.shape[0])/5):int(((i+1) * dftrain.shape[0])/5)]
        train_idx = list(set(index).difference(test_idx))

        fold_train = dftrain.iloc[train_idx]
        fold_test = dftrain.iloc[test_idx]

        for mgr in fold_train.groupby('manager_id'):
            # indices for this manager id
            test_mgr_idx = fold_test[fold_test.manager_id == mgr[0]].index

            # ratio of manager's listings that fall in each category
            dftrain.loc[test_mgr_idx, 'mgr_low_pct'] = (mgr[1].interest_level == 'low').mean()
            dftrain.loc[test_mgr_idx, 'mgr_med_pct'] = (mgr[1].interest_level == 'medium').mean()
            dftrain.loc[test_mgr_idx, 'mgr_high_pct'] = (mgr[1].interest_level == 'high').mean()

    # populate features for test data        
    dftest['mgr_low_pct'] = np.nan
    dftest['mgr_med_pct'] = np.nan
    dftest['mgr_high_pct'] = np.nan     

    for mgr in dftrain.groupby('manager_id'):
        # indices for this manager id
        test_mgr_idx = dftest[dftest.manager_id == mgr[0]].index

        # ratio of manager's listings that fall in each category
        dftest.loc[test_mgr_idx, 'mgr_low_pct'] = (mgr[1].interest_level == 'low').mean()
        dftest.loc[test_mgr_idx, 'mgr_med_pct'] = (mgr[1].interest_level == 'medium').mean()
        dftest.loc[test_mgr_idx, 'mgr_high_pct'] = (mgr[1].interest_level == 'high').mean()

    return dftrain, dftest