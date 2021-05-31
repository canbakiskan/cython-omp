
cdef extern from '/usr/include/x86_64-linux-gnu/cblas.h':
    void cblas_scopy(const int N, const float *X, const int incX,
                  float *Y, const int incY) nogil
    void cblas_saxpy(const int N, const float alpha, const float *X,
                  const int incX, float *Y, const int incY) nogil
    float  cblas_sdot(const int N, const float *X, const int incX,
                   const float *Y, const int incY) nogil