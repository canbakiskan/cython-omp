# from distutils.core import setup
# from distutils.extension import Extension
import numpy
# from Cython.Build import cythonize

from setuptools import Extension, setup
from Cython.Build import cythonize


extensions = [
    Extension("omp_cython", ["omp_cython.pyx"],
        include_dirs=[numpy.get_include(),"/usr/include/x86_64-linux-gnu/"],
        libraries=['cblas','lapack'],
        library_dirs=['/usr/lib/x86_64-linux-gnu/']
        ),
]
setup(
    ext_modules=cythonize(extensions),
    compiler_directives={'language_level' : "3"} 
)
