# learns the sparse dictionary by extracting patches from the train dataset

import numpy as np
from time import time
from sklearn.decomposition import MiniBatchDictionaryLearning
from .get_cifar import get_cifar10
import torch
import matplotlib.pyplot as plt


def extract_patches(images, patch_shape, stride, in_order="NHWC", out_order="NHWC"):
    assert images.ndim >= 2 and images.ndim <= 4
    if isinstance(images, np.ndarray):
        from sklearn.feature_extraction.image import _extract_patches

        if images.ndim == 2:  # single gray image
            images = np.expand_dims(images, 0)

        if images.ndim == 3:
            if images.shape[2] == 3:  # single color image
                images = np.expand_dims(images, 0)
            else:  # multiple gray images or single gray image with first index 1
                images = np.expand_dims(images, 3)

        elif in_order == "NCHW":
            images = images.transpose(0, 2, 3, 1)
        # numpy expects order NHWC
        patches = _extract_patches(
            images,
            patch_shape=(1, *patch_shape),
            extraction_step=(1, stride, stride, 1),
        ).reshape(-1, *patch_shape)
        # now patches' shape = NHWC

        if out_order == "NHWC":
            pass
        elif out_order == "NCHW":
            patches = patches.permute(0, 3, 1, 2)
        else:
            raise ValueError(
                'out_order not understood (expected "NHWC" or "NCHW")')

    elif isinstance(images, torch.Tensor):
        if images.ndim == 2:  # single gray image
            images = images.unsqueeze(0)

        if images.ndim == 3:
            if images.shape[2] == 3:  # single color image
                images = images.unsqueeze(0)
            else:  # multiple gray image
                images = images.unsqueeze(3)

        if in_order == "NHWC":
            images = images.permute(0, 3, 1, 2)
        # torch expects order NCHW

        patches = torch.nn.functional.unfold(
            images, kernel_size=patch_shape[:2], stride=stride
        )

        # all these operations are done to circumvent pytorch's N,C,W,H ordering

        patches = patches.permute(0, 2, 1)
        nb_patches = patches.shape[0] * patches.shape[1]
        patches = patches.reshape(nb_patches, patch_shape[2], *patch_shape[:2])
        # now patches' shape = NCHW
        if out_order == "NHWC":
            patches = patches.permute(0, 2, 3, 1)
        elif out_order == "NCHW":
            pass
        else:
            raise ValueError(
                'out_order not understood (expected "NHWC" or "NCHW")')

    return patches


def main():

    train_loader, _ = get_cifar10()

    dict_filepath = 'data/dictionary.npz'

    x_train = train_loader.dataset.data
    x_train = x_train / 255.0

    # Extract all patches
    print("Extracting reference patches...")
    t0 = time()

    print("Images shape: {}".format(x_train.shape))
    train_patches = extract_patches(
        x_train,
        (4, 4, 3),
        2,
        in_order="NHWC",
        out_order="NHWC",
    )
    print("Patches shape: {}".format(train_patches.shape))

    train_patches = train_patches.reshape(train_patches.shape[0], -1)

    print("done in %.2fs." % (time() - t0))

    print("Learning the dictionary...")
    t0 = time()
    dico = MiniBatchDictionaryLearning(
        n_components=500,
        alpha=1.0,
        n_iter=1000,
        batch_size=5,
        n_jobs=20,
    )
    # we employ column notation i.e. each column is an atom.
    # but sklearn uses row notation i.e. each row is an atom.
    # so what we call dictionary is their components_.transpose()

    dico.fit(train_patches)

    dictionary_transpose = dico.components_
    dt = time() - t0
    print("done in %.2fs." % dt)

    np.savez(dict_filepath, dict=dico.components_,
             params=dico.get_params())

    plt.figure(figsize=(11, 10))
    for i, atom in enumerate(dictionary_transpose[-400:]):
        plt.subplot(20, 20, i + 1)
        atom = atom.reshape(4, 4, 3)
        plt.imshow(
            (atom - atom.min()) / (atom.max() - atom.min()), interpolation="nearest"
        )

        # plt.axis("off")
        plt.xticks([])
        plt.yticks([])

    plt.suptitle("Dictionary learned from CIFAR10", fontsize=16)
    plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)

    fig_filepath = "dictionary_atoms.pdf"
    plt.savefig(fig_filepath)


if __name__ == "__main__":
    main()
