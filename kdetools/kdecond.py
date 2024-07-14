#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy.stats as st
from scipy._lib._util import check_random_state


class gaussian_kde(st.gaussian_kde):
    """Superclass of the `scipy.stats.gaussian_kde` class, adding
    conditional sampling and bandwidth selection by cross-validation."""
 
    def __init__(self, dataset, bw_method=None):
        """Create superclass of scipy gaussian_kde.
        """
        super(gaussian_kde, self).__init__(dataset, bw_method=bw_method)

    def _mvn_logpdf(self, x, mu, cov):
        """Vectorised evaluation of multivariate normal log-pdf for KDE.

        Evaluates log-density for all combinations of data points x and
        distribution means mu, assuming a fixed covariance matrix.

        Parameters
        ----------
        x : (m, n) ndarray
            Array of m n-dimensional values to evaluate.
        mu : (p, n) ndarray
            Array of p n-dimensional mean vectors to evaluate.
        cov : (n, n) ndarray
            Fixed covariance matrix.

        Returns
        -------
        logpdf : (m, p) ndarray
            Array of log-pdfs.
        """

        # Dimension of MVN
        mu = np.atleast_2d(mu)
        k = mu.shape[1]

        # Eigenvalues and eigenvectors of covariance matrix
        s, u = np.linalg.eigh(np.atleast_2d(cov))
        # Occasionally the smallest eigenvalue is negative
        s = np.abs(s) + np.spacing(1)

        # Terms in the log-pdf
        klog_2pi = k*np.log(2*np.pi)
        log_pdet = np.sum(np.log(s))

        # Mahalanobis distance using computed eigenvectors and eigenvalues
        maha = ((np.square((x[:,None] - mu) @ u)/s).sum(axis=2))
        logpdf = -0.5*(klog_2pi + log_pdet + maha)
        return logpdf

    def _mvn_pdf(self, x, mu, cov):
        """Vectorised evaluation of multivariate normal pdf for KDE."""
        return np.exp(self._mvn_logpdf(x, mu, cov))

    def kfold_split(self, X, k):
        """Lightweight k-fold CV function to avoid sklearn dependency."""
        splits = np.array_split(np.arange(X.shape[0]), k)
        return [(np.concatenate([splits[i] for i in set(range(k))-{j}]),
                 splits[j]) for j in range(k)]

    def set_bandwidth(self, bw_method=None, k=5, delta=1, n=101):
        """Add bandwidth selection by cross-validation.

        Parameters
        ----------
        bw_method : str, scalar or callable, optional
            As parent class, with extra 'cv' option.
        k : int, optional
            Number of folds in cross-validation.
        delta : float, optional
            Defines range of bandwidth grid in log space.
        n : int, optional
            Defines number of points in bandwidth grid.
        """

        if bw_method == 'cv':
            # Define trial bandwidths around initial guess of Silverman factor
            factor = self.silverman_factor()
            bws = np.exp(np.log(factor) + np.linspace(-delta, delta, n))

            # Cross-validation on log-likelihoods
            eps = np.spacing(1)
            x = self.dataset.T
            bw_lls = [np.mean([-np.log(self._mvn_pdf(x[j], x[i], bw**2*np.cov(x[i].T)
                                                    ).mean(axis=1)+eps).sum()
                               for i, j in self.kfold_split(x, k)])
                      for bw in bws]
            super(gaussian_kde, self).set_bandwidth(bw_method=bws[np.argmin(bw_lls)])
        else:
            super(gaussian_kde, self).set_bandwidth(bw_method=bw_method)

    def conditional_resample(self, size, x_cond, dims_cond, seed=None):
        """Fast conditional sampling of estimated pdf.
        
        Parameters
        ----------
        size : int
            Number of samples.
        x_cond : (m, n) ndarray
            Array of m n-dimensional values to condition on.
        dims_cond : (n,) int ndarray
            Indices of the dimensions which are conditioned on.
        seed : {None, int, `numpy.random.Generator`, `numpy.random.RandomState`}, optional
            Same behaviour as `kde.resample` method.

        Returns
        -------
        resample : (m, size, n) ndarray
            The sampled dataset.
        """

        # Check that dimensions are consistent
        x_cond = np.atleast_2d(x_cond.T).T
        if x_cond.shape[1] != len(dims_cond):
            print(f'Dimensions of x_cond {x_cond.shape} must be consistent '
                  f'with dims_cond ({len(dims_cond)})')
            return None

        random_state = check_random_state(seed)

        # Determine indices of dimensions to be sampled from
        dims_samp = np.array(list(set(range(self.d)) - set(dims_cond)))

        # Subset KDE kernel covariance matrix into blocks
        A = self.covariance[np.ix_(dims_samp, dims_samp)]
        B = self.covariance[np.ix_(dims_samp, dims_cond)]
        C = self.covariance[np.ix_(dims_cond, dims_cond)]

        # Evaluate log-densities at x_cond for all kernels
        logpdfs = self._mvn_logpdf(x_cond, self.dataset[dims_cond].T, C)

        # Convert to probabilities by correcting for precision then normalising
        pdfs = np.exp(logpdfs.T-logpdfs.max(axis=1))
        ps = (pdfs/pdfs.sum(axis=0)).T

        # Sample dataset kernels proportional to normalised pdfs at x_cond
        counts = np.array([random_state.multinomial(size, p) for p in ps])

        # Conditional mean and covariance matrices based on Schur complement
        BCinv = B @ np.linalg.inv(C)
        cov = A - BCinv @ B.T
        mus = np.swapaxes(self.dataset[dims_samp] +
                          BCinv @ (x_cond[:,:,None] - self.dataset[dims_cond]), 1, 2)

        # Sample from conditional kernel pdfs
        # Repeat means as many times as they were sampled in counts
        mus = np.repeat(mus.reshape(-1, dims_samp.size), counts.ravel(), axis=0
                        ).reshape(x_cond.shape[0], size, dims_samp.size)

        # As conditional covariance matrix is fixed, sample from zero mean mvn
        anoms = random_state.multivariate_normal(np.zeros(cov.shape[0]), cov,
                                                 size=(x_cond.shape[0], size))
        return mus + anoms
