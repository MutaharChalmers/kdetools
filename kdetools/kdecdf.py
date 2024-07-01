#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import scipy.stats as st


class kdecdf():
    def __init__(self, N=50, n_sigs=1, method='iqr'):
        """Efficient 1D Gaussian KDE modelling of empirical CDFs.
        Models fitted along axis 0 of a 2D numpy array.

        Parameters
        ----------
        N : int
            Number of points in 1D interpolation grid.
        n_sigs : int or float
            Number of kernel sigs beyond data bounds to buffer grid.
        method : str
            Method used to estimate kernel bandwidth. Both options based on the
            Silverman method, with the default `iqr` using the interquartile
            range method. For more details see:
            en.wikipedia.org/wiki/Kernel_density_estimation#Bandwidth_selection
        """

        self.N = N
        self.n_sigs = n_sigs
        self.method = method

    def fit(self, X):
        """Fit model to data.

        Parameters
        ----------
        X : (m, n) ndarray
            Data matrix.
        """

        if self.N is None:
            print('Model loaded from file - cannot be re-fit')
            return None

        X = np.atleast_2d(X.T).T

        # Calculate mins and maxes for each 1D vector and normalise
        self.mins = X.min(axis=0)
        self.maxs = X.max(axis=0)

        # Estimate bandwidth h using Silverman's factor
        n = X.shape[0]
        X_std = np.std(X, axis=0, ddof=1)
        if self.method == 'iqr':
            iqrs = np.diff(np.quantile(X, [0.25, 0.75], axis=0), axis=0)[0]
            self.sigs = 0.9*np.minimum(iqrs/1.34, X_std)*n**(-1/5)
        else:
            self.sigs = 1.06*X_std*n**(-1/5)

        # Calculate points at which to evaluate CDFs
        self.grids = np.linspace(self.mins-self.n_sigs*self.sigs, 
                                 self.maxs+self.n_sigs*self.sigs, self.N)

        # Calculate CDFs
        self.cdfs = st.norm.cdf((self.grids[:,None]-X)/self.sigs).mean(axis=1)

    def transform(self, X):
        """Calculate CDFs for data matrix X using fitted KDE model.

        Parameters
        ----------
        X : (m, n) ndarray
            Data matrix

        Returns
        -------
        U : (m, n) ndarray
            KDE-ECDF-transformed data matrix.
        """

        X = np.atleast_2d(X.T).T
        i = np.array([np.searchsorted(self.grids[:,k], X[:,k]) 
                            for k in range(X.shape[1])]).T
        j = np.arange(X.shape[1])[None,:]
        gradient = ((self.cdfs[i,j] - self.cdfs[i-1,j])/
                    (self.grids[i,j] - self.grids[i-1,j]))
        return self.cdfs[i-1,j] + gradient*(X-self.grids[i-1,j])

    def inverse(self, U):
        """Efficient 1D KDE estimates of ECDF along axis 0 of a 2D numpy array.
        
        Parameters
        ----------
        U : (m, n) ndarray
            Matrix of cumulative probabilities.

        Returns
        -------
        X : (m, n) ndarray
            Matrix of inverse KDE-ECDF-transformed data.
        """

        U = np.atleast_2d(U.T).T
        i = np.array([np.searchsorted(self.cdfs[:,k], U[:,k]) 
                      for k in range(U.shape[1])]).T
        j = np.arange(U.shape[1])[None,:]
        gradient = ((self.grids[i,j] - self.grids[i-1,j])/
                    (self.cdfs[i,j] - self.cdfs[i-1,j]))
        return self.grids[i-1,j] + gradient*(U-self.cdfs[i-1,j])

    def to_file(self, outpath, desc):
        """Save KDE model to file.
        
        Parameters
        ----------
        outpath : str
            Path to save model.
        desc : str
            Model description, and filename excluding suffix.
        """

        # Write numeric data to binary as a (2, m, n) array
        np.save(os.path.join(outpath, f'{desc}.npy'), 
                np.stack([self.grids, self.cdfs]))

    def from_file(self, inpath, desc):
        """Load KDE model from file.

        Parameters
        ----------
        inpath : str
            Path to saved model.
        desc : str
            Model description, and filename excluding suffix.
        """

        # Set None to flag to `fit()` method that this model cannot be refit
        self.N = None
        self.n_sigs = None

        # Read numeric data from binary
        self.grids, self.cdfs = np.load(os.path.join(outpath, f'{desc}.npy'))

    def calc_ecdf(self, X):
        """Calculate empirical CDF along axis 0 of 2D numpy array.

        Parameters
        ----------
        X : (m, n) ndarray
            Data matrix.

        Returns
        -------
        ecdf : (m, n) ndarray
            Calculated ECDFs.
        """

        return st.rankdata(X, axis=0)/X.shape[0]
