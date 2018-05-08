from distutils.core import setup, Extension
from Cython.Build import cythonize


cy_module = Extension("fastfood.fastfoodcy", ["fastfood/fastfoodcy.pyx", 
                                     "fastfood/fastfood.cc",
                                   ], 
                      include_dirs=['/Users/jonas/anaconda/envs/py36/include/eigen3/'], 
                      extra_compile_args = ['-O3',  
                                            '-fPIC', 
                                            '-march=native',
                                            '-shared', 
                                            '-std=c++11', 
                                            '-stdlib=libc++', # OSX only
                                            '-pedantic', 
                                            '-Wall',  
                                            '-Wshadow', 
                                            '-Wpointer-arith', 
                                            '-Wcast-qual',
                                            '-Wstrict-prototypes', 
                                            '-Wmissing-prototypes', 
                                            '-mavx', 
                                            '-g3'], 
                      language="c++", )
                      
setup(
    name='FastFood',
    version='0.1dev',
    packages=['fastfood',],
    license='BSD', 
    long_description=open('README.md').read(),
    ext_modules = cythonize([cy_module])
)
