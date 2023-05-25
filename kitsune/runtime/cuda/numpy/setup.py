from setuptools.command.build_ext import build_ext
from distutils.core import setup, Extension

import numpy
# TODO: This should point at the install path f
kitsune_lib_dir = "/projects/kitsune/x86_64/16.x/lib/"
def main():
        setup(name="kitrt_numpy_allocator",
              version="0.1.0",
              description="Kitsune runtime allocator for NumPy",
              author="Kitsune",
              author_email="kitsune@lanl.gov",
              ext_modules=[Extension(
                           name="kitrt",
                           sources=['kitrt.c'],
                           include_dirs=[numpy.get_include()],
                           library_dirs=[kitsune_lib_dir],
                           libraries=['kitrt','LLVM'],
                           runtime_library_dirs=[kitsune_lib_dir])],
              platforms=['Linux'],
              install_requires=['numpy>=1.22.0'],
              zip_safe = False,
              python_requires='>=3.8')
        
if __name__ == "__main__":
        main()
