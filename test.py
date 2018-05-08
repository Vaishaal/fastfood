import ff_python_call
import sklearn.kernel_approximation
from sklearn.metrics.pairwise import rbf_kernel
import numpy as np
import math
from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original', data_home="/tmp/mnist")
X = mnist.data[:2048,:].astype('float32')
X = np.hstack((X, np.zeros((X.shape[0], 2048 - X.shape[1])))).astype('float32')
np.random.seed(0)
g = np.random.randn(8192).astype('float32')
D = 8
M = 8192 
phase = np.random.uniform(low=0.0, high=2*math.pi,size=M)
sigma = 2048
phi_fastfood = np.sqrt(2/M) * np.cos(ff_python_call.fastfood(g,X,M, scale=sigma) + phase)
K_fastfood = phi_fastfood.dot(phi_fastfood.T)
K = -(2 * X.dot(X.T))
K += (np.linalg.norm(X,axis=1)**2)[:, np.newaxis]
K += (np.linalg.norm(X,axis=1)**2)[:, np.newaxis].T
K /= -(2.0*sigma*sigma)
K = np.exp(K, K)
print(K_fastfood)
print(K)
max_err = np.max(np.abs(K - K_fastfood))
print("Max err", max_err)
