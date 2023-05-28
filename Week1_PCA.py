import numpy as np

def pca_scratch(X_stand, k):
    """
    X_stand: standardized data
    k: number of principal components
    """

    """ TODO: 
    Part 2.1:
        - implement PCA
    """
# Find Covariance
    cov = np.cov(X_stand.T)

    # Find eigenvalues, eigenvectors
    eig_vals, eig_vecs = np.linalg.eig(cov)

    eig_vecs_sorted = eig_vecs[:, np.argsort(-eig_vals)[:k]]

    # project the data onto the new feature space, dot product to project onto eigenvalues
    X_pca = X_stand @ eig_vecs_sorted

    # variance
    var = eig_vals/np.sum(eig_vals)

    return X_pca, var

# @STUDENT: do not change the {import}s below!  
from library import pca_scratch
import numpy as np
from sklearn import preprocessing

def variance(data, k):
    """
    data: the data (not standardized)
    k: the number of principal components
    """
    # number of samples
    N = len(data)

    # initialize total variance for each sample
    var = np.empty(N)

    """ TODO: 
    Part 2.4:
        - compute the total variance covered with k components for each sample
    """
    for i in range(N): 
        sampledata_standardized = preprocessing.StandardScaler(
    ).fit_transform(np.array(data[f"Sample{i+1}"]))
        X_pca, variance_array = pca_scratch(sampledata_standardized, k)
        var[i] = np.sum(variance_array[:k])
    return var