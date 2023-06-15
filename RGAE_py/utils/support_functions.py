import numpy as np
import utils
from torchvision import transforms
from sklearn.preprocessing import MinMaxScaler
from utils import plot_det_map

def get_auc_training(model, data_tensor, ground_truth):
    '''
        data_tensor: Must have same dims as training data
    '''

    # data_tensor = model.data_tensor
    m = model.m
    n = model.n

    data_input = data_tensor[:,:].clone().transpose(1,2)
    data_output, z_output = model(data_input)

    input_np = data_tensor.clone().detach().numpy()[0,:,:]
    input_np = input_np.reshape((m,n,-1))
    output_np = data_output.clone().detach().numpy()[0,:,:].transpose(1, 0)
    output_np = output_np.reshape((m,n,-1))

    det_map=np.sum((input_np-output_np)**2, axis=2)

    roc_auc_scores = utils.ROC(det_map, ground_truth, 0)

    # plot_det_map(input_np[:,:,0], model.y, det_map)

    return roc_auc_scores

def get_auc_training_int(model, data_tensor1, data_tensor2, ground_truth, alpha):
    '''
        data_tensor: Must have same dims as training data
    '''

    # data_tensor = model.data_tensor
    m = model.m
    n = model.n

    data_input1 = data_tensor1[:,:].clone().transpose(1,2)
    data_input2 = data_tensor2[:,:].clone().transpose(1,2)
    
    data_output, z_output = model(data_input1, data_input2)

    input_np = model.alpha * data_tensor1.clone().detach().numpy()[0,:,:] + (1 - alpha) * data_tensor2.clone().detach().numpy()[0,:,:]
    input_np = input_np.reshape((m,n,-1))

    min_input = np.min(input_np)
    max_input = np.max(input_np)

    input_np = (input_np - min_input)/(max_input - min_input)

    output_np = data_output.clone().detach().numpy()[0,:,:].transpose(1, 0)
    output_np = output_np.reshape((m,n,-1))

    det_map=np.sum((input_np-output_np)**2, axis=2)

    roc_auc_scores = utils.ROC(det_map, ground_truth, 0)

    # plot_det_map(input_np[:,:,0], model.y, det_map)

    return roc_auc_scores

def get_auc_training_rgvae(model, data_tensor, ground_truth):

    # data_tensor = model.data_tensor
    m = model.m
    n = model.n

    data_input = data_tensor[:,:].clone().transpose(1,2)
    data_output, z_mean, z_logvar = model(data_input)

    input_np = data_tensor.clone().detach().numpy()[0,:,:]
    input_np = input_np.reshape((m,n,-1))
    output_np = data_output.clone().detach().numpy()[0,:,:].transpose(1, 0)
    output_np = output_np.reshape((m,n,-1))

    det_map=np.sum((input_np-output_np)**2, axis=2)

    roc_auc_scores = utils.ROC(det_map, ground_truth, 0)

    # plot_det_map(input_np[:,:,0], model.y, det_map)

    return roc_auc_scores

def calculate_det_map(input_data, output_data, m, n):

    new_X = input_data.clone().detach().numpy()[0,:,:]
    new_X = new_X.reshape((m,n,-1))

    output_data_np = output_data.clone().detach().numpy()[0,:,:].transpose(1, 0)
    new_output_data = output_data_np.reshape((m,n,-1))

    det_map=np.sum((new_X-new_output_data)**2, axis=2)

    return det_map, new_X

def calculate_det_map_int(input_data1, input_data2, output_data, m, n, alpha):

    new_X1 = input_data1.clone().detach().numpy()[0,:,:]
    new_X1 = new_X1.reshape((m,n,-1))

    new_X2 = input_data2.clone().detach().numpy()[0,:,:]
    new_X2 = new_X2.reshape((m,n,-1))

    new_X = alpha * new_X1 + (1 - alpha) * new_X2
    new_X = (new_X - np.min(new_X)) / (np.max(new_X) - np.min(new_X))

    output_data_np = output_data.clone().detach().numpy()[0,:,:].transpose(1, 0)
    new_output_data = output_data_np.reshape((m,n,-1))

    det_map=np.sum((new_X-new_output_data)**2, axis=2)

    return det_map, new_X

def transform_to_tensor(X):

    [m,n,L] = np.shape(X)
    X_2D = X.reshape((m*n, L))

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    X_2D = X_2D.astype('float32')
    scaler = MinMaxScaler()
    X_2D = scaler.fit_transform(X_2D)
    data_tensor = transform(X_2D)

    return data_tensor

