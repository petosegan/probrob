from distutils.core import setup
from Cython.Build import cythonize

setup(
            ext_modules = cythonize("probrob/ray_trace.pyx")
            )
