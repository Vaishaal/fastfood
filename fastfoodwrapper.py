# coding: utf-8
import ctypes
from numpy.ctypeslib import ndpointer
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state

ff = ctypes.cdll.LoadLibrary('fastfood.so').fastfood
ff.argtypes = [ndpointer(ctypes.c_float), ndpointer(ctypes.c_float), ndpointer(ctypes.c_float), ndpointer(ctypes.c_float), ndpointer(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_int]

def _is_power_2(num):
    return (num != 0 and ((num & (num - 1)) == 0))

# def fastfood(gaussian, samples, outsize, seed=0, scale=1):
#     ''' gaussian Input should be IID  N(0,1) gaussian
#         variance is captured in scale (which is (1/sigma*sqrt(d)))
#     '''

#     assert(_is_power_2(outsize))
#     gaussian = gaussian.astype('float32')
#     # np.random.seed(seed)
#     radamacher = np.random.binomial(1, 0.5, outsize).astype('float32')
#     radamacher[np.where(radamacher== 0)] = -1
#     chisquared = np.sqrt(np.random.chisquare(outsize, outsize)).astype('float32') * 1.0/np.linalg.norm(gaussian).astype('float32')
#     out = np.zeros((outsize, samples.shape[0]), 'float32', order="F")
#     ff(gaussian, radamacher, chisquared, samples, out, outsize, samples.shape[1], samples.shape[0])
#     normalizer = (scale/np.sqrt(samples.shape[1]))
#     out /= normalizer
#     return out.T


def __fastfood(gaussian, radamacher, chisquared, samples, output, outsize, insize, numsamples):
    ''' LOW LEVEL WRAPPER
    This function will mutate output '''
    ff(gaussian, radamacher, chisquared, samples, output, outsize, insize, numsamples)

def npot(x):
    return int(2**np.ceil(np.log2(x)))


class FastFood(BaseEstimator, TransformerMixin):
    def __init__(self, scale=1, n_components=100, 
                 random_state = None):
        self.scale = scale
        self.random_state = random_state
        self.n_components = n_components
        self.gaussian = None

    def fit(self, X, y=None):
        if (X.shape[1] >= self.n_components) :
            raise ValueError("New array must have more features than input data")

        random_state = check_random_state(self.random_state)

        self.gaussian = random_state.normal(0, 1, self.n_components).astype(np.float32).reshape(-1, 1)

        self.radamacher = random_state.binomial(1, 0.5, self.n_components).astype('float32')
        self.radamacher[np.where(self.radamacher== 0)] = -1
        self.chisquared = np.sqrt(random_state.chisquare(self.n_components, self.n_components)).astype('float32') * 1.0/np.linalg.norm(self.gaussian).astype('float32')
        
        self.radamacher = random_state.binomial(1, 0.5, self.n_components).astype('float32')
        self.radamacher[np.where(self.radamacher== 0)] = -1
        self.chisquared = np.sqrt(random_state.chisquare(self.n_components, self.n_components)).astype('float32') * 1.0/np.linalg.norm(self.gaussian).astype('float32')

        self.phase_offsets = random_state.uniform(0, 2 * np.pi, size=self.n_components).astype(np.float32)
        return self


    
    def transform(self, X):
        # pad X out to POT
        X_pad = np.zeros((X.shape[0], npot(X.shape[1])), dtype=np.float32)
        X_pad[:, :X.shape[1]] = X
        X = X_pad

        if (X.shape[1] >= self.n_components) :
            raise ValueError("n_components must be larger than next_pow_of_two(X.shape[1])")

        out = np.zeros((self.n_components, X.shape[0]), 'float32', order="F")
        ff(self.gaussian, self.radamacher, self.chisquared, X, out, self.n_components, 
           X.shape[1], X.shape[0])
        normalizer = (self.scale/np.sqrt(X.shape[1]))
        out /= normalizer

        phi_fastfood = np.sqrt(2/self.n_components) * np.cos(out.T + self.phase_offsets)
        return phi_fastfood

        # X = check_array(X)
        # print(self.gaussian.shape, self.radamacher.shape, self.chisquared.shape, X.shape, self.n_components)
        # res = fastfood(self.gaussian, self.radamacher, self.chisquared, 
        #                X, self.n_components, self.random_state, 
        #                scale=self.scale)

        # return np.cos(res + self.random_offset) / np.sqrt(4*npot(X.shape[1]))
 

