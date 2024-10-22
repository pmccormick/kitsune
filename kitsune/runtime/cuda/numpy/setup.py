from setuptools.command.build_ext import build_ext
from distutils.core import setup, Extension
import os
import numpy

kitsune_install_prefix = "/projects/kitsune/x86_64/18.x"
path = os.environ.get("KITSUNE_INSTALL_PREFIX", kitsune_install_prefix)
kitsune_lib_dir = path + "/lib/clang/18/lib/"

CUDA_PATH = os.environ.get("CUDA_PATH")
if not CUDA_PATH:
    CUDA_PATH = os.environ.get("CUDA_HOME")
if not CUDA_PATH:
    raise RuntimeError('Environment variable CUDA_HOME or CUDA_PATH is not set')
print('cuda install prefix: ', CUDA_PATH)
print('kitsune install lib dir: ', kitsune_lib_dir)

include_path_list = [numpy.get_include(),"../"] + [os.path.join(CUDA_PATH, 'include')]
print(include_path_list)
# TODO: This should point at the install path
def main():
        setup(name="kitrt_numpy_allocator",
              version="0.1.0",
              description="Kitsune runtime allocator for NumPy",
              author="Kitsune",
              author_email="kitsune@lanl.gov",
              ext_modules=[Extension(
                           name="kitrt",
                           sources=['kitrt.c'],
                           include_dirs=include_path_list,
                           library_dirs=[kitsune_lib_dir],
                           libraries=['kitrt','LLVM'],
                           runtime_library_dirs=[kitsune_lib_dir])],
              platforms=['Linux'],
              install_requires=['numpy>=1.22.0'],
              zip_safe = False,
              python_requires='>=3.8')
        
if __name__ == "__main__":
        main()
