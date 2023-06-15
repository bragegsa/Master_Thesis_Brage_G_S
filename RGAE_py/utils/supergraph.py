import numpy as np
from skimage.segmentation import slic
from sklearn.decomposition import PCA

def supergraph(data, S):
    M, N, L = data.shape
    X = data.reshape(M*N, L)
    
    # PCA
    pca = PCA(n_components=1)
    Y = pca.fit_transform(X)
    Y = Y.reshape(M,N,1)
    y = Y[:,:,-1]

    y = (y - np.min(y)) / (np.max(y) - np.min(y))
    
    # Superpixels
    labels = slic(y.reshape(M, N), n_segments=S)
    nums = np.max(labels)
    
    # Construct Laplacian Matrix
    W = np.zeros((M*N, M*N))
    spec2 = 4
    X_new = []
    idex = []
    cnt = 0
    
    for num in range(1, nums+1):
        idx = np.where(labels == num)[0]
        K = len(idx)
        x = X[idx, :]

        if num == 1:
            X_new = x
        else:
            X_new = np.vstack((X_new, x))

        idex = np.concatenate((idex, idx))
        
        tmp = np.zeros((K, K))
        for i in range(K):
            s = x[i, :]
            for j in range(i, K):
                tmp[i, j] = np.exp(-np.sum((s - x[j, :])**2) / (2 * spec2))
        W[cnt:cnt+K, cnt:cnt+K] = tmp
        cnt += K

    D = np.diag(np.sum(W, axis=0))
    SG = D - W
    
    return SG, X_new, idex
