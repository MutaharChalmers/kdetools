#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy.linalg as sl
import scipy.optimize as so
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

    def silverman_factor_ref(self):
        """Compute the (refined) Silverman factor.

        Returns
        -------
        s : ndarray
            The Silverman factor(s).
        """
        iqrs = np.diff(np.quantile(self.dataset, [0.25, 0.75], axis=1), axis=0)
        return (0.9*np.minimum(iqrs/1.34, self.dataset.std(ddof=1, axis=1))
                *np.power(self.neff, -1/5)
               ).ravel()/self.dataset.std(ddof=1, axis=1)

    def set_bandwidth(self, bw_method=None, bw_type='diagonal', k=None):
        """Add bandwidth selection by cross-validation.

        Parameters
        ----------
        bw_method : str, scalar or callable, optional
            As parent class, with extra 'cv' option.
        bw_type : str, optional
            Type of bandwidth matrix. Options are `diagonal`,
            `covariance` and `equal`.
        k : int, optional
            Number of folds in cross-validation. Leave One Out by default.
        """

        if bw_method == 'cv':
            # Define bandwidth log-likelihood functions for cross-validation
            # Vector of factors scaling each dimension with no cross-covariance
            if bw_type == 'diagonal':
                h0 = self.dataset.std(ddof=1, axis=1) * self.silverman_factor_ref()
                def negloglike(h, Xeval, Xfit):
                    return -np.log(self._mvn_pdf(Xeval, Xfit, np.diag(h**2)
                                                ).mean(axis=1)).sum()
            # Single factor scaling data covariance matrix
            elif bw_type == 'covariance':
                h0 = self.silverman_factor_ref().mean()
                def negloglike(h, Xeval, Xfit):
                    return -np.log(self._mvn_pdf(Xeval, Xfit, np.cov(Xfit.T)*h**2
                                                ).mean(axis=1)).sum()
            # Single factor scaling in all dimensions with no cross-covariance
            elif bw_type == 'equal':
                h0 = self.dataset.std(ddof=1) * self.silverman_factor_ref().mean()
                def negloglike(h, Xeval, Xfit):
                    return -np.log(self._mvn_pdf(Xeval, Xfit, np.eye(self.d)*h**2
                                                ).mean(axis=1)).sum()
            else:
                print('bw_type must be diagonal, covariance or equal')
                return None
            self.bw_type = bw_type

            # Define cross-validation - default LOOCV
            if k is None:
                k = self.dataset[0].size
            splits = self.kfold_split(self.dataset.T, k)

            # Minimise negative log-likelihood CV
            def negloglike_cv(h, X, splits):
                return np.mean([negloglike(h, X[j], X[i]) for i, j in splits])
            res = so.minimize(negloglike_cv, h0, args=(self.dataset.T, splits),
                              method='nelder-mead')
            self.h = res['x']
            self.loglike_cv = -res['fun']
            self.covariance_factor = lambda: self.h
            self._compute_covariance()
        else:
            self.bw_type = 'covariance_original'
            super(gaussian_kde, self).set_bandwidth(bw_method=bw_method)

    def _compute_covariance(self):
        """Computes the covariance matrix for each Gaussian kernel using
        covariance_factor(). Custom version for bw_method='cv'.
        """

        # With bw_method='cv', factor can be a vector of values
        self.factor = self.covariance_factor()

        # Cache covariance and Cholesky decomp of covariance
        if not hasattr(self, '_data_cho_cov'):
            self._data_covariance = np.atleast_2d(np.cov(self.dataset, rowvar=1,
                                                  bias=False, aweights=self.weights))
            self._data_cho_cov = sl.cholesky(self._data_covariance, lower=True)

        if self.bw_type == 'diagonal':
            self.covariance = np.diag(self.factor**2)
        elif self.bw_type == 'equal':
            self.covariance = np.eye(self.d)*self.factor**2
        else: # 'covariance' or 'covariance_original'
            self.covariance = self._data_covariance * self.factor**2

        self.cho_cov = sl.cholesky(self.covariance, lower=True).astype(np.float64)
        self.log_det = 2*np.log(np.diag(self.cho_cov * np.sqrt(2*np.pi))).sum()

    @property
    def inv_cov(self):
        """Custom for bw_method='cv' to handle possibly vector factor.
        Just don't change the dataset attribute, as the CV-derived
        bandwidths will not update on ruuning covariance_factor."""
        self.factor = self.covariance_factor()
        if self.bw_type == 'diagonal':
            return np.linalg.inv(np.diag(self.factor**2))
        elif self.bw_type == 'equal':
            return np.linalg.inv(np.eye(self.d)*self.factor**2)
        else: # 'covariance' or 'covariance_original'
            return np.linalg.inv(self._data_covariance * self.factor**2)

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
        dims_samp = np.setdiff1d(range(self.d), dims_cond)

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