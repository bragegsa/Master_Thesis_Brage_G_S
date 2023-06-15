import torch
from utils import ROC, RGAE_int_MK, train_RGAE_int, plot_loss, plot_det_map, calculate_det_map_int, plot_auc_and_epochloss
import os
import scipy.io as sio
import time

data_path = os.path.join(os.getcwd(),'parameters/')
param_file_name = 'default'
dataset_list = sio.loadmat(os.path.join(data_path, param_file_name))['dataset_list']
lr_list = sio.loadmat(os.path.join(data_path, param_file_name))['lr_list'][0]
lambda_list = sio.loadmat(os.path.join(data_path, param_file_name))['lambda_list'][0]

at = '_def'
grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
total_result = []

for alpha in grid:
    results_name = 'name_of_results' + str(alpha)
    list_of_auc = []

    for i, dataset_name in enumerate(dataset_list):

        dataset_name = dataset_name.replace('.mat', '')
        dataset_name = dataset_name.replace(' ', '')
        print('Training on dataset:', dataset_name)

        hidden_dim = 100
        model = RGAE_int_MK(dataset_name, at, hidden_dim)

        model.alpha = alpha

        model.learning_rate = lr_list[i]
        model.lambda_g = lambda_list[i]
        model.epochs = 500
        model.batch_num = 100
        model.batch_size = int(model.N/model.batch_num)

        start_time = time.perf_counter()
        model, epochs_loss, losses, auc_list = train_RGAE_int(model, es_bool=True)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        epochs_loss_np = [loss.detach().numpy() for loss in epochs_loss] 

        data_tensor1 = model.data_tensor1
        data_tensor2 = model.data_tensor2

        data_input1 = data_tensor1[:,:].clone().transpose(1,2)
        data_input2 = data_tensor2[:,:].clone().transpose(1,2)
        data_output, z_output = model(data_input1, data_input2)

        det_map, new_X = calculate_det_map_int(data_tensor1, data_tensor2, data_output, model.m, model.n, model.alpha)
        roc_auc_scores = ROC(det_map, model.y, 0)

        print("Finished training on:", dataset_name)
        print('Final AUC score:', dataset_name, 'is:', round(roc_auc_scores,4))
        print('Total training time:', round(elapsed_time,2), '\n')
        
        folder_name = 'results/' + dataset_name + '/'

        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        sio.savemat((folder_name + '/' + results_name + '.mat'), {'auc': roc_auc_scores, 'y': det_map, 'run_time': elapsed_time, 'epochs_loss': epochs_loss_np, 'list_of_auc': auc_list})

        list_of_auc.append(round(roc_auc_scores,4))

    print('--- FINAL AUC SCORES: --- \n')
    print(','.join([str(auc) for auc in list_of_auc]))
    print('\n')

    total_result.append(','.join([str(auc) for auc in list_of_auc]))

for i in range(len(total_result)):

    print(str(grid[i]) + ":", total_result[i])