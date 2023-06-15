import torch
import numpy as np

def RGAE_mat_loss(output, target, encoded, lambda_g, L, spectral_bands, version='1'):
    '''
        This function returns a variant of the loss translated from the matlab interpretation of
        the paper "Hyperspectral Anomaly Detection With RobustGraph Autoencoders"
    '''
    
    # version = '1'

    if version == '1':
        error = output - target
        graph_loss = torch.matmul(encoded, torch.from_numpy(L + L.T).type(encoded.dtype))
        # print('Graph loss:', lambda_g * torch.mean(graph_loss))   
        # print('Reconstruction loss:', torch.mean(error ** 2))     
        loss = torch.mean(error ** 2) + lambda_g * torch.mean(graph_loss)

    elif version == '2':
        error = output - target
        error_sq = error**2

        res = torch.sum(error_sq, dim=0)

        two_res = 2*res
        reciprocal_two_res = 1/two_res
        D = reciprocal_two_res.repeat(spectral_bands, 1)

        grad = error*D
        grad_mean = torch.mean(grad)

        graph_loss = torch.matmul(encoded, torch.from_numpy(L + L.T).type(encoded.dtype))
        graph_loss_mean = torch.mean(graph_loss)

        loss = grad_mean + lambda_g * graph_loss_mean
    elif version == '3':
        error = output - target
        graph_loss = torch.matmul(encoded, torch.from_numpy(L + L.T).type(encoded.dtype))
        # print('Graph loss:', lambda_g * torch.mean(graph_loss))   
        # print('Reconstruction loss:', torch.mean(error ** 2))   
        loss = torch.mean(error ** 2) + lambda_g * torch.mean(graph_loss)
        # loss = torch.mean(error ** 2)

    return loss

def interpolation_loss(output, input1, input2, encoded, lambda_g, L, alpha):

    error1 = output - input1
    error2 = output - input2
    graph_loss = torch.matmul(encoded, torch.from_numpy(L + L.T).type(encoded.dtype))     
    loss = alpha * torch.mean(error1 ** 2) + (1 - alpha) * torch.mean(error2 ** 2) + lambda_g * torch.mean(graph_loss)

    return loss

def interpolation_loss2(output, input1, input2, encoded, lambda_g, L1, L2, alpha):

    error1 = output - input1
    error2 = output - input2
    graph_loss1 = torch.matmul(encoded, torch.from_numpy(L1 + L1.T).type(encoded.dtype))     
    graph_loss2 = torch.matmul(encoded, torch.from_numpy(L2 + L2.T).type(encoded.dtype))     
    loss = alpha * (torch.mean(error1 ** 2) + lambda_g * torch.mean(graph_loss1)) + (1 - alpha) * (torch.mean(error2 ** 2) + lambda_g * torch.mean(graph_loss2))

    return loss

def RGAE_paper_loss(output, target, encoded, lambda_g, L, N, spectral_bands):
    '''
        This function returns the loss as computed in the paper "Hyperspectral Anomaly Detection 
        With RobustGraph Autoencoders"
    '''

    error = output - target
    L_tensor = torch.from_numpy(L).type(encoded.dtype)

    # l21_norm = torch.sum(torch.sqrt(torch.sum(error**2, dim=1)))/(2 * N)
    l21_norm = torch.sum(torch.sqrt(torch.sum(error**2, dim=1)))/(2 * spectral_bands * N)
    # tr = lambda_g/N * torch.trace(torch.matmul(torch.matmul(encoded, L_tensor), encoded.transpose(0,1)))
    tr = lambda_g/(N**2) * torch.trace(torch.matmul(torch.matmul(encoded, L_tensor), encoded.transpose(0,1)))
    loss2 = l21_norm + tr

    return loss2

def RGAE_MSE_loss(output, target, encoded, lambda_g, L):

    criterion = torch.nn.MSELoss()
    mse_loss = criterion(output, target)

    graph_loss = torch.matmul(encoded, torch.from_numpy(L + L.T).type(encoded.dtype))    
    loss = mse_loss + lambda_g * torch.mean(graph_loss)

    return loss

def early_stopping(epochs_loss, n_loss, std_limit):
    '''
        epochs_loss: List of all losses per epoch
        n_loss: the number of epochs to check for convergence
        std_limit: the minimal standard deviation for early stopping
    '''

    epochs_loss_np = [loss.detach().numpy() for loss in epochs_loss] 
    
    # Check if the number of elements are over the convergance
    if len(epochs_loss) > n_loss and epochs_loss_np[-1] < 0.2:
        std_epochs = np.std(epochs_loss_np[-n_loss:])

        if std_epochs < std_limit:

            print('Early stopped training as oss is convereged. Last loss:', epochs_loss_np[-1])
            
            return True

    return False

def early_stopping_2(epochs_loss, patience=50, min_delta=0.005):
    """Check if the epochs_loss list is converging for the last patience elements.
    
    Args:
        epochs_loss (list): A list of loss values for each epoch.
        patience (int): The number of epochs to wait before stopping if convergence is detected.
        min_delta (float): The minimum change in the mean of the last patience elements to be considered convergence.
    
    Returns:
        bool: True if convergence is detected, False otherwise.
    """
    if len(epochs_loss) < patience:
        return False
    
    last_losses = epochs_loss[-patience:]
    mean_loss = sum(last_losses) / patience
    variance = sum((x - mean_loss) ** 2 for x in last_losses) / patience
    std_dev = variance ** 0.5
    
    if std_dev < min_delta:
        return True
    
    return False
