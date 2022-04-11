try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from setuptools.command.build_ext import build_ext
import numpy

# Extensions
# pykdtree (kd tree)
pykdtree = Extension(
    'im2mesh.utils.libkdtree.pykdtree.kdtree',
    sources=[
        'im2mesh/utils/libkdtree/pykdtree/kdtree.c',
        'im2mesh/utils/libkdtree/pykdtree/_kdtree_core.c'
    ],
    language='c',
    extra_compile_args=['-std=c99', '-O3', '-fopenmp'],
    extra_link_args=['-lgomp'],
)

ext_modules = [
    pykdtree,
]

setup(
    ext_modules = cythonize(ext_modules),
    include_dirs=[numpy.get_include()],
    cmd_class={
        'build_ext': build_ext
    }
)