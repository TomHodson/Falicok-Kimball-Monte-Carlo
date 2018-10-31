from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    name="Falicov Kimball Monte Carlo",
    ext_modules=cythonize("*.pyx", include_path = [numpy.get_include()], annotate=True),
)
