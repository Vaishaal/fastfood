from cython.view cimport array as cvarray
import numpy as np

cdef extern from "fastfood.h":
   cdef void fastfood(float* gaussian,
                 float* radamacher,
                 float* chiSquared,
                 float* patchMatrix,
                 float* output,
                 int outSize,
                 int inSize,
                 int numPatches)


def foosum(x):
    print(100)


def foo(x):
    print("hello", x)


def ffood(float [:] gaussian, 
       float [:] radamacher, 
       float [:] chiSquared, 
       float [:, :] patchMatrix, 
       float [:, :] output):

    fastfood(&gaussian[0], &radamacher[0], &chiSquared[0], 
             &patchMatrix[0, 0], &output[0, 0], 
             output.shape[0], 
             patchMatrix.shape[1], 
             patchMatrix.shape[0])

    return 0


