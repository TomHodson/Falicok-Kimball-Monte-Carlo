from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension("*", ["*.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args = ['-ip_no_inlining'], #required to disable ipo so that it will build with ICC
        #libraries=[...],
        #library_dirs=[...]
        ),

]

setup(
    name="Falicov Kimball Monte Carlo",
    py_modules=['jobmanager'],
    ext_modules=cythonize(extensions, annotate=True),
    install_requires=['Click'],
    entry_points='''
        [console_scripts]
        run_mcmc=jobmanager:run_mcmc
    ''',
)



#command to build inplace is: python setup.py build_ext --inplace
#command to install is: pip install . --editable
