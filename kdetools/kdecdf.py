#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import scipy.stats as st
import scipy.special as ss
from .kde import gaussian_kde


class kdecdf():
    def __init__(self, N=50, buffer_bws=1, method='silverman', nanfill=None):
        """Efficient 1D Gaussian KDE modelling of empirical CDFs.
        Models fitted along axis 0 of a 2D numpy array.

        Parameters
        ----------
        N : int, optional
            Number of points in 1D interpolation grid.
        buffer_bws : int or float, optional
            Number of kernel bandwidths beyond data bounds to buffer grid.
        method : str, optional
            Method used to estimate kernel bandwidth. Option are 'silverman'
            (Silverman rule of thumb using the interquartile range method),
            'scott' (Scott's rule of thumb), 'cv' (leave one out CV) and
            'precomputed' (bandwidths generated elsewhere). See:
            en.wikipedia.org/wiki/Kernel_density_estimation#Bandwidth_selection
        nanfill : str, optional
            Fill nans, and if so, how to do it. None doesn't fill nans, 'mean'
            and 'median' fills nans with the mean and median along axis 0.
            In the fit method, nan filling is always done after calculating
            bandwidths, to ensure a well-defined ECDF is calculated.
        """

        self.N = N
        self.buffer_bws = buffer_bws
        self.method = method
        self.nanfill = nanfill

    def fit(self, X, min_bw=1e-18, bws=None):
        """Fit model to data.

        Parameters
        ----------
        X : (m, n) ndarray or DataFrame
            Data matrix.
        min_bw : float, optional
            Minimum bandwidth. Defaults to 1e-18.
        """

        if self.N is None:
            print('Model loaded from file - cannot be re-fit')
            return None

        # Save or define column names and convert to 2D ndarray for processing
        if isinstance(X, pd.DataFrame):
            self.columns = X.columns
        elif isinstance(X, pd.Series):
            self.columns = X.name
        else:
            self.columns = np.arange(X.shape[1])
        X = np.atleast_2d(X.T).T

        # Calculate mins and maxes for each 1D vector and normalise
        self.mins = np.nanmin(X, axis=0)
        self.maxs = np.nanmax(X, axis=0)

        # Estimate bandwidth
        m, n = X.shape
        X_std = np.nanstd(X, axis=0, ddof=1)
        if self.method.lower() == 'silverman':
            iqrs = np.diff(np.nanquantile(X, [0.25, 0.75], axis=0), axis=0)[0]
            self.bws = 0.9*np.minimum(iqrs/1.34, X_std)*m**(-1/5)
        elif self.method.lower() == 'scott':
            self.bws = 1.06*X_std*m**(-1/5)
        elif self.method.lower() == 'cv':
            bws = np.zeros(n)
            for i in range(n):
                if np.abs(np.ptp(X[:,i])) < np.spacing(1):
                    bws[i] = np.spacing(1)
                else:
                    kdeobj = gaussian_kde(X[:,i])
                    kdeobj.set_bandwidth(bw_method='cv')
                    bws[i] = kdeobj.factor[0]
            self.bws = bws * X_std
        elif self.method.lower() == 'precomputed':
            self.bws = bws
        else:
            print('Method must be silverman, scott, cv or precomputed')
            return None

        # Clip zero bandwidths to target minimum
        self.bws = np.clip(self.bws, min_bw, None)

        # Handle nans by appending row of nans to grids and cdfs
        nanfill = np.empty(n)
        nanfill[:] = np.nan

        # Calculate points at which to evaluate CDFs
        grids = np.linspace(self.mins-self.buffer_bws*self.bws,
                            self.maxs+self.buffer_bws*self.bws, self.N)
        self.grids = np.vstack([grids, nanfill])

        # Calculate CDFs; fill nans with median, i.e. assuming few nans
        X = np.where(np.isnan(X), np.nanmedian(X, axis=0), X)
        self.cdfs = ss.ndtr((self.grids[:,None]-X)/self.bws).mean(axis=1)

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
        if self.nanfill == 'median':
            X = np.where(np.isnan(X), np.nanmedian(X, axis=0), X)
        elif self.nanfill == 'mean':
            X = np.where(np.isnan(X), np.nanmean(X, axis=0), X)
        else:
            pass

        i = np.array([np.searchsorted(self.grids[:,k], X[:,k])
                            for k in range(X.shape[1])]).T
        j = np.arange(X.shape[1])[None,:]
        gradient = ((self.cdfs[i,j] - self.cdfs[i-1,j])/
                    (self.grids[i,j] - self.grids[i-1,j]))
        return self.cdfs[i-1,j] + gradient*(X-self.grids[i-1,j])

    def inverse(self, U):
        """KDE quantile function for [0, 1] uniform matrix U.

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
        if self.nanfill == 'median':
            U = np.where(np.isnan(U), np.nanmedian(U, axis=0), U)
        elif self.nanfill == 'mean':
            U = np.where(np.isnan(U), np.nanmean(U, axis=0), U)
        else:
            pass

        i = np.array([np.searchsorted(self.cdfs[:,k], U[:,k])
                      for k in range(U.shape[1])]).T
        j = np.arange(U.shape[1])[None,:]
        gradient = ((self.grids[i,j] - self.grids[i-1,j])/
                    (self.cdfs[i,j] - self.cdfs[i-1,j]))
        return self.grids[i-1,j] + gradient*(U-self.cdfs[i-1,j])

    def to_file(self, outpath, desc, format='parquet'):
        """Save KDE model to file.

        Parameters
        ----------
        outpath : str
            Path to save model.
        desc : str
            Model description, and filename excluding suffix.
        format : str, optional
            File format to use. Parquet by default, otherwise numpy binary.
        """

        if format == 'parquet':
            grids = pd.DataFrame(self.grids, columns=self.columns)
            cdfs = pd.DataFrame(self.cdfs, columns=self.columns)
            pd.concat({'grids': grids, 'cdfs':cdfs}
                      ).to_parquet(os.path.join(outpath, f'{desc}.parquet'))
        else:
            # Write numeric data to binary as a (2, m, n) array
            np.save(os.path.join(outpath, f'{desc}.npy'),
                    np.stack([self.grids, self.cdfs]))

    def from_file(self, inpath, desc, format='parquet'):
        """Load KDE model from file.

        Parameters
        ----------
        inpath : str
            Path to saved model.
        desc : str
            Model description, and filename excluding suffix.
        format : str, optional
            File format to use. Parquet by default, otherwise numpy binary.
        """

        # Set None to flag to `fit()` method that this model cannot be refit
        self.N = None
        self.buffer_bws = None

        if format == 'parquet':
            df = pd.read_parquet(os.path.join(inpath, f'{desc}.parquet'))
            self.columns = df.columns
            self.grids = df.loc['grids'].to_numpy()
            self.cdfs = df.loc['cdfs'].to_numpy()
        else:
            # Read numeric data from binary
            self.grids, self.cdfs = np.load(os.path.join(inpath, f'{desc}.npy'))

    def calc_ecdf(self, X, axis=0):
        """Calculate empirical CDF along axis of ndarray.

        Parameters
        ----------
        X : ndarray
            Data matrix.
        axis : int, optional
            Axis along which to calculate ECDF.

        Returns
        -------
        ecdf : ndarray
            Calculated ECDFs.
        """

        return st.rankdata(X, axis=axis)/X.shape[axis]
