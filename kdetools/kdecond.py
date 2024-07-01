#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy.stats as st
from scipy._lib._util import check_random_state


class gaussian_kde(st.gaussian_kde):
    """Superclass of the `scipy.stats.gaussian_kde` class, adding
    conditional sampling functionality."""
 
    def __init__(self, dataset, bw_method=None):
        """Create superclass of scipy gaussian_kde.
        """
        super(gaussian_kde, self).__init__(dataset, bw_method=bw_method)

    def conditional_resample(self, size, x_cond, dims_cond, seed=None):
        """Conditional sampling of estimated pdf.

        Use Schur complement to evaluate conditional kernels.
        
        Parameters
        ----------
        size : int
            Number of samples.
        x_cond : 1D list or array
            Values to condition on.
        dims_cond : 1D list or array of ints
            Indices of the dimensions which are conditioned.
        seed : {None, int, `numpy.random.Generator`, `numpy.random.RandomState`}, optional
            Same behaviour as `kde.resample` method.

        Returns
        -------
        resample : (self.d, `size`) ndarray
            The sampled dataset.
        """

        # Preparations
        x_cond = np.array(x_cond)
        random_state = check_random_state(seed)

        # Determine indices of dimensions to be sampled from
        dims_samp = np.array(list(set(range(self.d)) - set(dims_cond)))

        # Subset full KDE covariance matrix into blocks
        A = self.covariance[np.ix_(dims_samp, dims_samp)]
        B = self.covariance[np.ix_(dims_samp, dims_cond)]
        C = self.covariance[np.ix_(dims_cond, dims_cond)]

        # Evaluate densities at x_cond for all kernels
        densities = np.array([st.multivariate_normal(mu, C).pdf(x_cond)
                              for mu in self.dataset[dims_cond].T])
        
        # Sample indices of data points proportional to normalised pdfs at x_cond
        ixs = random_state.choice(densities.size, size=size, p=densities/densities.sum())
    
        # Count sampling frequency of each data point
        counts = np.bincount(ixs, minlength=densities.size)

        # Conditional mean and covariance matrices for each data point using
        BCinv = B @ np.linalg.inv(C)
        cov = A - BCinv @ B.T
        mus = self.dataset[dims_samp] + BCinv @ (x_cond[:,None] - self.dataset[dims_cond])

        # Sample from conditional kernel pdfs 
        samples = [random_state.multivariate_normal(mu, cov, size=n)
                   for n, mu in zip(counts, mus.T)]
        return np.concatenate(samples)
