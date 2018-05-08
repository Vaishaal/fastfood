from distutils.core import setup, Extension

# INCLUDES=-I/Users/jonas/anaconda/envs/py36/include/eigen3/
# CPPFLAGS:= -O3 -fPIC  -march=native -shared -std=c++0x -pedantic -Wall -Wshadow -Wpointer-arith -Wcast-qual -Wstrict-prototypes -Wmissing-prototypes -mavx -g3 -larmadillo -I/data/vaishaal/eigen/ $(INCLUDES)

ff_module = Extension('fastfood',
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
                      sources = ['fastfood/fastfood.cc'])
setup(
    name='FastFood',
    version='0.1dev',
    packages=['fastfood',],
    license='BSD', 
    long_description=open('README.md').read(),
    ext_modules = [ff_module]
)
