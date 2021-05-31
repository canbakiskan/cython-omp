import numpy as np
from numba import njit, float32, int32, int64


@njit(fastmath=True)
def random_OMP(patches, phi, nb_ompcomp, top_T):
    nb_patches = patches.shape[0]
    reconstructions = np.zeros_like(patches)

    for i in range(nb_patches):
        patch = patches[i, :]
        residual = patch
        coeffs = np.zeros((nb_ompcomp), dtype=np.float64)
        indices = np.zeros((nb_ompcomp), dtype=np.int64)
        for j in range(nb_ompcomp):
            inners = phi.T@residual
            if j == 0:
                if top_T == 1:
                    indices[j] = np.argmax(np.abs(inners))
                else:
                    indices[0] = np.random.choice(np.argsort(
                        np.abs(inners))[-top_T:], 1)
                coeffs[0] = inners[indices[0]]
            else:
                indices[j] = np.argmax(np.abs(inners))
                (coeffs[:(j+1)], _, _,
                 _) = np.linalg.lstsq(phi[:, indices[:(j+1)]], patch, rcond=None)
            residual = patch-phi[:, indices[:(j+1)]]@coeffs[:(j+1)]
            if np.max(np.abs(residual)):
                break
        reconstructions[i, :] = patch-residual

    return reconstructions


@njit(fastmath=True)
def OMP_batch_numba(patches, phi, nb_ompcomp):
    nb_patches = patches.shape[0]
    reconstructions = np.empty_like(patches)

    coeffs = np.empty((nb_ompcomp), dtype=np.float32)
    indices = np.empty((nb_ompcomp), dtype=np.int32)
    for i in range(nb_patches):
        patch = patches[i, :]
        residual = patch
        for j in range(nb_ompcomp):
            inners = phi.T@residual
            indices[j] = np.argmax(np.abs(inners))
            if j == 0:
                coeffs[0] = inners[indices[0]]
            else:
                coeffs[:(j+1)] = np.linalg.lstsq(phi[:,
                                                     indices[:(j+1)]], patch, rcond=-1)[0]
            residual = patch-phi[:, indices[:(j+1)]]@coeffs[:(j+1)]
            # if np.max(np.abs(residual)) < 1e-6:
            #     break
        reconstructions[i, :] = patch-residual

    return reconstructions


@njit(fastmath=True)
def OMP_numba(patch, phi, nb_ompcomp):
    reconstruction = np.zeros_like(patch)
    residual = patch
    coeffs = np.zeros((nb_ompcomp), dtype=np.float32)
    indices = np.zeros((nb_ompcomp), dtype=np.int32)
    for j in range(nb_ompcomp):
        inners = phi.T@residual
        if j == 0:
            indices[0] = np.argmax(np.abs(inners))
            coeffs[0] = inners[indices[0]]
        else:
            indices[j] = np.argmax(np.abs(inners))
            coeffs[:(j+1)] = np.linalg.lstsq(phi[:, indices[:(j+1)]], patch)[0]
        residual = patch-phi[:, indices[:(j+1)]]@coeffs[:(j+1)]
        if np.max(np.abs(residual)) < 1e-6:
            break
    reconstruction = patch-residual

    return reconstruction
