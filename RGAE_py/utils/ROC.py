import numpy as np
import matplotlib.pyplot as plt

def ROC(result, map, display):
    """
    Calculate the AUC value of result.

    Args:
    - result: detection map (rows by columns).
    - map: groundtruth (rows by columns).
    - display: display the ROC curve if display==1.

    Returns:
    - auc: AUC values of 'y'.
    """

    M, N = result.shape
    p_f = np.zeros(M*N)    # false alarm rate
    p_d = np.zeros(M*N)    # detection probability
    ind = np.argsort(result.flatten())[::-1]
    res = np.zeros(M*N)
    map = map.flatten()
    N_anomaly = np.sum(map)
    N_pixel = M*N
    N_miss = 0

    for i in range(M*N):
        res[ind[i]] = 1
        N_detected = np.sum(res*map)
        if map[ind[i]] == 0:
            N_miss += 1
        p_f[i] = N_miss/(N_pixel-N_anomaly)
        p_d[i] = N_detected/N_anomaly

    auc = np.trapz(p_d, p_f)    # calculate the AUC value

    if display==1:
        plt.plot(p_f, p_d)
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC Curve')
        plt.show()

    return auc
