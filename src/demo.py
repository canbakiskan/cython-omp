import numpy as np
from numba import njit, float32, int32
from numpy.lib.function_base import diff
import ray
from .get_cifar import get_cifar10
from .learn_patch_dict import extract_patches
from tqdm import tqdm
from time import time
import torch
import matplotlib.pyplot as plt
from . import reconstruction_methods as methods
# from .reconstruction_methods.lasso import lasso_batch_numba, lasso_numba
# from .reconstruction_methods.lasso_cython import lasso_batch_cython, lasso_cython
# from .reconstruction_methods.omp import OMP_batch_numba, OMP_numba
# from .reconstruction_methods.omp_cython import OMP_batch_cython, OMP_cython
import argparse


def combine_patches(patches, image_shape, stride, in_order="NHWC", out_order="NHWC"):
    numpy = False
    if isinstance(patches, np.ndarray):
        numpy = True
        patches = torch.from_numpy(patches)

    if in_order == "NHWC":
        patches = patches.permute(0, 3, 1, 2)
    # torch expects order NCHW

    patch_shape = patches.shape[1:]
    n_w = (image_shape[0] - patch_shape[1]) // stride + 1
    n_h = (image_shape[1] - patch_shape[2]) // stride + 1
    nb_patch_per_img = n_w * n_h
    nb_imgs = patches.shape[0] // nb_patch_per_img
    patches = patches.reshape(nb_imgs, nb_patch_per_img, -1).permute(0, 2, 1)

    images = torch.nn.functional.fold(
        patches, output_size=image_shape[:2], kernel_size=patch_shape[1:], stride=stride
    )

    divisor = torch.nn.functional.unfold(
        torch.ones_like(images), kernel_size=patch_shape[1:], stride=stride
    )

    divisor = torch.nn.functional.fold(
        divisor, output_size=image_shape[:2], kernel_size=patch_shape[1:], stride=stride
    )

    images /= divisor

    if out_order == "NHWC":
        images = images.permute(0, 2, 3, 1)
    elif out_order == "NCHW":
        pass
    else:
        raise ValueError(
            'out_order not understood (expected "NHWC" or "NCHW")')

    if numpy:
        images = images.numpy()

    return images


def normalize_diff_img(diff_img):
    abs_max = np.max(np.abs(diff_img))
    diff_img /= abs_max
    diff_img *= 0.5
    diff_img += 0.5

    return diff_img


@ray.remote(num_returns=1)
def ray_function(method, patches, i, batch_size, dictionary):
    if "lasso" in method.__name__:
        args = [dictionary, np.asfortranarray(
            patches[i:i+batch_size].squeeze()), 1.0, 500]

    elif "OMP" in method.__name__:
        args = [patches[i:i+batch_size].squeeze(), dictionary, 5]
    else:
        raise Exception('Method name should have either lasso or OMP in it.')

    return method(*args)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--method",
        type=str,
        default="OMP_batch_cython",
        choices=[
            "OMP_numba",
            "OMP_batch_numba",
            "OMP_cython",
            "OMP_batch_cython",
            "lasso_numba",
            "lasso_batch_numba",
            "lasso_cython",
            "lasso_batch_cython"
        ]
    )

    args = parser.parse_args()

    # reconstruction_method = globals()[args.method]
    reconstruction_method = getattr(methods, args.method)

    file = np.load('data/dictionary.npz', allow_pickle=True)
    dictionary = file["dict"].T
    dictionary = np.array(dictionary, order="C", dtype=np.float32)

    nb_images = 1000  # total is 50000
    nb_patches = 15*15*nb_images
    batch_size = 100 if "batch" in args.method else 1  # 5000 is the optimal

    train_loader, _ = get_cifar10()
    images = train_loader.dataset.data[:nb_images].astype(np.float32)
    images = images / 255.0
    patches = extract_patches(images, (4, 4, 3), 2)
    patches = patches.reshape(patches.shape[0], -1)

    ray.init(num_cpus=40)
    patches_id = ray.put(patches)
    dictionary_id = ray.put(dictionary)

    start = time()
    result_ids = [ray_function.remote(reconstruction_method, patches_id, i, batch_size, dictionary_id)
                  for i in range(0, nb_patches, batch_size)]

    reconstructed_patches = np.array(
        ray.get(result_ids)).reshape(-1, 4, 4, 3)

    end = time()

    reconstructions = combine_patches(reconstructed_patches, (32, 32, 3), 2)
    reconstructions = reconstructions.clip(0, 1)

    print(f"Time per patch: {(end-start)/nb_patches}")
    print(f"Total time: {(end-start)}")

    plt.figure(figsize=(3, 10))
    for i in range(10):
        plt.subplot(10, 3, 3*i+1)
        plt.imshow(images[i])
        plt.xticks([])
        plt.yticks([])

        plt.subplot(10, 3, 3*i+2)
        plt.imshow(reconstructions[i])
        plt.xticks([])
        plt.yticks([])

        plt.subplot(10, 3, 3*i+3)
        plt.imshow(normalize_diff_img(images[i]-reconstructions[i]))
        plt.xticks([])
        plt.yticks([])

    plt.tight_layout()
    plt.savefig(f'{args.method}_reconstructions.pdf')


if __name__ == "__main__":
    main()
