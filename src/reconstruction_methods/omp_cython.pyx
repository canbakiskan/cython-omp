#!python
# cython: embedsignature=True, binding=True

cimport cython
cimport numpy as np
import numpy as np
from libc.math cimport fabs
cimport omp_cython_header
from libc.stdio cimport printf

ctypedef enum CBLAS_ORDER: 
    CblasRowMajor=101,
    CblasColMajor=102

ctypedef enum CBLAS_TRANSPOSE:
    CblasNoTrans=111,
    CblasTrans=112,
    CblasConjTrans=113,
    AtlasConj=114
    
def OMP_batch_cython(const float[:,::1] patches, const float[:,::1] phi, const int nb_ompcomp):
    return np.asarray(OMP_batch_cython_cdef(patches, phi, nb_ompcomp))

def OMP_cython(const float[:,::1] patch, const float[:,::1] phi, const int nb_ompcomp):
    return np.asarray(OMP_cython_cdef(patch, phi, nb_ompcomp))


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef OMP_batch_cython_cdef(const float[:,::1] patches, const float[:,::1] phi, const int nb_ompcomp):
    cdef: 
        int nb_patches = patches.shape[0]
        int patch_size = patches.shape[1]
        int nb_atoms = phi.shape[1]
        float[:,::1] reconstructions = np.empty_like(patches)
        # float[::1,:] reconstructions = np.empty((patches.shape[0],patches.shape[1]), order="F", dtype=np.float32)
        const float* patch
        float[:] residual = np.empty((patch_size), dtype=np.float32)
        float[:] coeffs = np.zeros((patch_size), dtype=np.float32)
        float[:] inners = np.zeros((nb_atoms), dtype=np.float32)
        int[:] indices = np.zeros((nb_ompcomp), dtype=np.int32)
        float[::1,:] relevant_atoms = np.empty((patch_size, nb_ompcomp), dtype=np.float32, order="F")
        CBLAS_TRANSPOSE transpose = CblasTrans
        CBLAS_TRANSPOSE no_transpose = CblasNoTrans
        CBLAS_ORDER row_major = CblasRowMajor # C order
        CBLAS_ORDER column_major = CblasColMajor # F orders
        
    omp_cython_header.cblas_scopy(nb_patches*patch_size, &patches[0,0], 1, &reconstructions[0,0], 1)
    
    for i in range(nb_patches):
        omp_cython_header.cblas_scopy(patch_size, &patches[i, 0], 1, &residual[0], 1)

        for j in range(nb_ompcomp):
            omp_cython_header.cblas_sgemv(row_major, transpose, patch_size, nb_atoms,
                1.0, &phi[0,0], nb_atoms, &residual[0], 1, 0.0, &inners[0], 1)
            # inners = phi.T@residual
            indices[j]=omp_cython_header.cblas_isamax(nb_atoms, &inners[0], 1)

            for k in range(j+1):
                omp_cython_header.cblas_scopy(patch_size, &phi[0,indices[k]], nb_atoms, &relevant_atoms[0,k], 1)
            # indices[j] = np.argmax(np.abs(inners))

            if j == 0:
                coeffs[0] = inners[indices[0]]
            else:
                omp_cython_header.cblas_scopy(patch_size, &patches[i, 0], 1, &coeffs[0], 1)
                omp_cython_header.clapack_sgels(column_major, no_transpose,  patch_size, j+1, 
                    1, &relevant_atoms[0,0], patch_size, &coeffs[0], patch_size)
            # coeffs[:(j+1)] = np.linalg.lstsq(phi[:, indices[:(j+1)]], patch, rcond=-1)[0]

            for k in range(j+1):
                omp_cython_header.cblas_scopy(patch_size, &phi[0,indices[k]], nb_atoms, &relevant_atoms[0,k], 1)
            
            omp_cython_header.cblas_scopy(patch_size, &patches[i, 0], 1, &residual[0], 1)
            omp_cython_header.cblas_sgemv(column_major, no_transpose, patch_size, j+1,
                -1.0, &relevant_atoms[0,0], patch_size, &coeffs[0], 1, 1.0, &residual[0], 1)
            # residual = patch-phi[:, indices[:(j+1)]]@coeffs[:(j+1)]

        for j in range(patch_size):
            reconstructions[i, j]-=residual[j]
        # reconstructions[i, :] = patch-residual

    return reconstructions


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef OMP_cython_cdef(const float[:,::1] patches, const float[:,::1] phi, const int nb_ompcomp):
    #BURAYI DOLDUR BURASI KOPYA HEP
    cdef: 
        int nb_patches = patches.shape[0]
        int patch_size = patches.shape[1]
        int nb_atoms = phi.shape[1]
        float[:,::1] reconstructions = np.empty_like(patches)
        # float[::1,:] reconstructions = np.empty((patches.shape[0],patches.shape[1]), order="F", dtype=np.float32)
        const float* patch
        float[:] residual = np.empty((patch_size), dtype=np.float32)
        float[:] coeffs = np.zeros((patch_size), dtype=np.float32)
        float[:] inners = np.zeros((nb_atoms), dtype=np.float32)
        int[:] indices = np.zeros((nb_ompcomp), dtype=np.int32)
        float[::1,:] relevant_atoms = np.empty((patch_size, nb_ompcomp), dtype=np.float32, order="F")
        CBLAS_TRANSPOSE transpose = CblasTrans
        CBLAS_TRANSPOSE no_transpose = CblasNoTrans
        CBLAS_ORDER row_major = CblasRowMajor # C order
        CBLAS_ORDER column_major = CblasColMajor # F orders
        
    omp_cython_header.cblas_scopy(nb_patches*patch_size, &patches[0,0], 1, &reconstructions[0,0], 1)
    
    for i in range(nb_patches):
        omp_cython_header.cblas_scopy(patch_size, &patches[i, 0], 1, &residual[0], 1)

        for j in range(nb_ompcomp):
            omp_cython_header.cblas_sgemv(row_major, transpose, patch_size, nb_atoms,
                1.0, &phi[0,0], nb_atoms, &residual[0], 1, 0.0, &inners[0], 1)
            # inners = phi.T@residual
            indices[j]=omp_cython_header.cblas_isamax(nb_atoms, &inners[0], 1)

            for k in range(j+1):
                omp_cython_header.cblas_scopy(patch_size, &phi[0,indices[k]], nb_atoms, &relevant_atoms[0,k], 1)
            # indices[j] = np.argmax(np.abs(inners))

            if j == 0:
                coeffs[0] = inners[indices[0]]
            else:
                omp_cython_header.cblas_scopy(patch_size, &patches[i, 0], 1, &coeffs[0], 1)
                omp_cython_header.clapack_sgels(column_major, no_transpose,  patch_size, j+1, 
                    1, &relevant_atoms[0,0], patch_size, &coeffs[0], patch_size)
            # coeffs[:(j+1)] = np.linalg.lstsq(phi[:, indices[:(j+1)]], patch, rcond=-1)[0]

            for k in range(j+1):
                omp_cython_header.cblas_scopy(patch_size, &phi[0,indices[k]], nb_atoms, &relevant_atoms[0,k], 1)
            
            omp_cython_header.cblas_scopy(patch_size, &patches[i, 0], 1, &residual[0], 1)
            omp_cython_header.cblas_sgemv(column_major, no_transpose, patch_size, j+1,
                -1.0, &relevant_atoms[0,0], patch_size, &coeffs[0], 1, 1.0, &residual[0], 1)
            # residual = patch-phi[:, indices[:(j+1)]]@coeffs[:(j+1)]

        for j in range(patch_size):
            reconstructions[i, j]-=residual[j]
        # reconstructions[i, :] = patch-residual

    return reconstructions

