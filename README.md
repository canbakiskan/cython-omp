# Cython Implementation of Orthogonal Matching Pursuit

This repository implements orthogonal matching pursuit (OMP) and least absolute shrinkage and selection operator (lasso) algorithm in numba and Cython.

## Requirements

Install the requirements:

```bash
sudo apt-get install libblas-dev liblapack-dev # or equivalent
sudo apt-get install libatlas-base-dev # or equivalent
pip install -r requirements.txt
```

Then add current directory to `$PYTHONPATH`:

```bash
export PYTHONPATH="${PYTHONPATH}:/path/containing/project/directory"
```

Following commands assume the name of the folder is `cython_omp`.

## Dictionary Learning

To learn the overcomplete dictionary, run:

```bash
python -m cython_omp.src.learn_patch_dict
```

## Compiling Cython scripts

```bash
cd src/reconstruction_methods
python omp_setup.py build_ext --inplace
python lasso_setup.py build_ext --inplace
```

## Reconstruction Demo

```bash
python -m cython_omp.src.demo
```

## License

Apache License 2.0
