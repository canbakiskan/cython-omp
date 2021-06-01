#!python
# cython: embedsignature=True, binding=True

cimport cython
cimport numpy as np
import numpy as np
from libc.math cimport fabs
cimport lasso_cython_header

cdef float fsign(float f) nogil:
    if f == 0:
        return 0
    elif f > 0:
        return 1.0
    else:
        return -1.0


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef float[:] ct_solver(int n_samples, int n_features,
                           const float[::1, :] X, const float[:] y,
                           const float[:] norms_X_col,
                           float[:] beta, float[:] residual,
                           float alpha,
                           int maxiter=200) nogil:
    """Coordinate descent solver for l2 reg kl interpolation."""

    cdef:
        int inc = 1
        float tmp
        float mbetaj

    lasso_cython_header.cblas_scopy(n_samples, &y[0], inc, &residual[0], inc)
    for i in range(maxiter):
        maxw = 0.
        for j in range(n_features):
            # tmp is the prox argument
            if beta[j] != 0.:
                lasso_cython_header.cblas_saxpy(n_samples, beta[j], &X[0, j], inc, &residual[0], inc)
                #residual += X[:, j] * beta[j]

            #tmp = X[:, j].dot(residual)
            tmp = lasso_cython_header.cblas_sdot(n_samples, &residual[0], inc, &X[0, j], inc)
            # l1 thresholding
            beta[j] = fsign(tmp) * max(fabs(tmp) - alpha, 0) / norms_X_col[j]

            if beta[j] != 0.:
                mbetaj = - beta[j]
                lasso_cython_header.cblas_saxpy(n_samples, mbetaj, &X[0, j], inc, &residual[0], inc)
                # residual += - beta[j] * X[:, j]

    return beta


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def lasso_cython(const float[::1, :] X, const float[:] y, float alpha, int maxiter=200):

    cdef:
        int n_samples
        int n_features
        int inc = 1
    n_samples = X.shape[0]
    n_features = X.shape[1]

    dtype = np.float32
    cdef:
        float[:] sol
        float[:] residual = np.empty(n_samples, dtype=dtype)
        float[:] beta = np.zeros(n_features, dtype=dtype)
        float[:] norms_X_col = np.empty(n_features, dtype=dtype)
    # compute norms_X_col
    for j in range(n_features):
        norms_X_col[j] = lasso_cython_header.cblas_sdot(n_samples, &X[0, j], inc, &X[0, j], inc)
    with nogil:
        sol = ct_solver(n_samples, n_features, X, y, norms_X_col, beta, residual, alpha, maxiter)

    lasso_cython_header.cblas_saxpy(n_samples, -1, &y[0], inc, &residual[0], inc)

    return -np.asarray(residual)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def lasso_batch_cython(const float[::1, :] X, const float[::1,:] Y, float alpha, int maxiter=200):
    ### BROKEN!!! DONT KNOW WHERE 
    cdef:
        int n_samples
        int n_features
        int inc = 1
    n_samples = X.shape[0]
    n_features = X.shape[1]

    dtype = np.float32
    cdef:
        float[:] sol
        float[:] residual = np.empty(n_samples, dtype=dtype)
        float[:] beta = np.zeros(n_features, dtype=dtype)
        float[:] norms_X_col = np.empty(n_features, dtype=dtype)
        float[::1,:] reconstructions = np.empty((Y.shape[0], n_samples),order="F", dtype=dtype,)
        float[:] y = np.empty(n_samples, dtype=dtype)
    
    for k in range(Y.shape[0]):
        beta = np.zeros(n_features, dtype=dtype)
        lasso_cython_header.cblas_scopy(n_samples, &Y[k,0], inc, &y[0], inc)
        # y=Y[k]
        # compute norms_X_col
        for j in range(n_features):
            norms_X_col[j] = lasso_cython_header.cblas_sdot(n_samples, &X[0, j], inc, &X[0, j], inc)
        with nogil:
            sol = ct_solver(n_samples, n_features, X, y, norms_X_col, beta, residual, alpha, maxiter)

        lasso_cython_header.cblas_saxpy(n_samples, -1, &y[0], inc, &residual[0], inc)
        lasso_cython_header.cblas_scopy(n_samples, &residual[0], inc, &reconstructions[k,0], inc)


    return -np.asarray(reconstructions)

