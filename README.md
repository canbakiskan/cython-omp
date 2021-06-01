# Cython Implementation of Orthogonal Matching Pursuit

This repository implements orthogonal matching pursuit (OMP) and least absolute shrinkage and selection operator (lasso) algorithm in numba and Cython.

## Requirements

Install the requirements:

```bash
sudo apt-get install libatlas-base-dev # or equivalent
pip install -r requirements.txt
```

Then add project and reconstruction methods directories to `$PYTHONPATH`:

```bash
export PYTHONPATH="${PYTHONPATH}:/path/containing/project/directory/"
export PYTHONPATH="${PYTHONPATH}:/path/containing/project/directory/cython-omp/src/reconstruction_methods/" # this is because of ray's relative import inability
```

Following commands assume the name of the folder is `cython-omp`.

## Dictionary Learning

To learn the overcomplete dictionary, run:

```bash
python -m cython-omp.src.learn_patch_dict
```

## Compiling Cython scripts

```bash
cd src/reconstruction_methods
python omp_setup.py build_ext --inplace
python lasso_setup.py build_ext --inplace
```

## Reconstruction Demo

```bash
python -m cython_omp.src.demo --method=<method>
```

choices for the method are: 
- OMP_numba
- OMP_batch_numba
- OMP_cython
- OMP_batch_cython (default)
- lasso_numba
- lasso_batch_numba
- lasso_cython
- lasso_batch_cython

## License

Apache License 2.0
