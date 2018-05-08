import fastfoodwrapper
import sklearn.kernel_approximation
from sklearn.metrics.pairwise import rbf_kernel
import numpy as np
import math
from sklearn.datasets import fetch_mldata
import sklearn.metrics.pairwise


def standalone_test():

    mnist = fetch_mldata('MNIST original', data_home="/tmp/mnist")
    X = mnist.data[:2048,:].astype('float32')
    X = np.hstack((X, np.zeros((X.shape[0], 2048 - X.shape[1])))).astype('float32')


    np.random.seed(0)

    M = 8192
    g = np.random.randn(M).astype('float32')
    #D = 8

    phase = np.random.uniform(low=0.0, high=2*math.pi,size=M)
    sigma = 2048.0
    phi_fastfood = np.sqrt(2/M) * np.cos(fastfoodwrapper.fastfood(g,X,M, scale=sigma) + phase)
    K_fastfood = phi_fastfood.dot(phi_fastfood.T)

    gamma = 1.0/(2.0*sigma**2)
    K = sklearn.metrics.pairwise.rbf_kernel(X, X, gamma =  gamma)

    # K = -(2 * X.dot(X.T))
    # K += (np.linalg.norm(X,axis=1)**2)[:, np.newaxis]
    # K += (np.linalg.norm(X,axis=1)**2)[:, np.newaxis].T
    # K /= -(2.0*sigma*sigma)
    # K = np.exp(K, K)
    print(K_fastfood)
    print(K)
    max_err = np.max(np.abs(K - K_fastfood))
    print("Max err", max_err)

def sklearn_test():

    mnist = fetch_mldata('MNIST original', data_home="/tmp/mnist")
    X = mnist.data[:2048,:].astype('float32')
    X = np.hstack((X, np.zeros((X.shape[0], 2048 - X.shape[1])))).astype('float32')


    np.random.seed(0)

    M = 8192
    g = np.random.randn(M).astype('float32')
    #D = 8

    phase = np.random.uniform(low=0.0, high=2*math.pi,size=M)
    sigma = 1024.0

    ffw = fastfoodwrapper.FastFood(scale=sigma, 
                                   n_components = M)

    #phi_fastfood = np.sqrt(2/M) * np.cos(fastfoodwrapper.fastfood(g,X,M, scale=sigma) + phase)
    phi_fastfood = ffw.fit_transform(X)

    K_fastfood = phi_fastfood.dot(phi_fastfood.T)

    gamma = 1.0/(2.0*sigma**2)
    K = sklearn.metrics.pairwise.rbf_kernel(X, X, gamma =  gamma)

    # K = -(2 * X.dot(X.T))
    # K += (np.linalg.norm(X,axis=1)**2)[:, np.newaxis]
    # K += (np.linalg.norm(X,axis=1)**2)[:, np.newaxis].T
    # K /= -(2.0*sigma*sigma)
    # K = np.exp(K, K)
    print(K_fastfood)
    print(K)
    max_err = np.max(np.abs(K - K_fastfood))
    print("Max err", max_err)

if __name__ == "__main__":
    sklearn_test()
