cdef extern from '/usr/include/x86_64-linux-gnu/clapack.h':
    int clapack_sgels(int Order, int Tz, const int M, 
                    const int N, const int Nrhs, float *A,
                    const int lda, float *B, const int ldb) nogil


cdef extern from '/usr/include/x86_64-linux-gnu/cblas.h':

    void cblas_scopy(const int N, const float *X, const int incX,
                    float *Y, const int incY) nogil

    void cblas_sgemv(const int Order, const int TransA, const int M,
                    const int N, const float alpha, const float *A, 
                    const int lda, const float *X, const int incX,
                    const float beta, float *Y, const int incY) nogil

    int cblas_isamax(const int N, const float  *X, const int incX) nogil