from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

import os

os.environ["CC"] = "icc"
os.environ["LDSHARED"] = "icc -shared"

extensions = [
    Extension("*", ["*.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args = ['-ip_no_inlining', '-xhost', '-qopenmp'],
        extra_link_args = [],
        libraries=[],
        library_dirs=[],
        ),

]

setup(
    name="Falicov Kimball Monte Carlo",
    py_modules=['jobmanager'],
    ext_modules=cythonize(extensions, annotate=True, language_level=3),
    install_requires=['Click'],
    entry_points='''
        [console_scripts]
        run_mcmc=jobmanager:run_mcmc_command
    ''',
)



#command to build inplace is: python setup.py build_ext --inplace
#command to install is: pip install --editable .
