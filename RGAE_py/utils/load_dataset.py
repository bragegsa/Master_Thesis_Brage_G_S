import scipy.io as sio
import os

def load_dataset(dataset_mat, dataset_var, folder_name):
    '''
        KPCA dataset variables: abu
        Default dataset variables: data
    '''
    data_path = os.path.join(os.getcwd(),'datasets', folder_name)
    data_path_map = os.path.join(os.getcwd(),'datasets/default')

    X = sio.loadmat(os.path.join(data_path, dataset_mat))[dataset_var]
    y = sio.loadmat(os.path.join(data_path_map, dataset_mat))['map']

    return X, y

def load_graph(dataset_name, at):

    data_path = os.path.join(os.getcwd(),'graphs/',dataset_name)

    filename = dataset_name + at
    SG = sio.loadmat(os.path.join(data_path, filename))['SG']
    X_new = sio.loadmat(os.path.join(data_path, filename))['X_new']
    idx = sio.loadmat(os.path.join(data_path, filename))['idx']

    return SG, X_new, idx

def load_loss_auc(dataset_name, filename):

    data_path = os.path.join(os.getcwd(),'results/',dataset_name)
    auc_list = sio.loadmat(os.path.join(data_path, filename))['list_of_auc'][0,:]
    epochs_loss = sio.loadmat(os.path.join(data_path, filename))['epochs_loss'][0,:]

    return auc_list, epochs_loss