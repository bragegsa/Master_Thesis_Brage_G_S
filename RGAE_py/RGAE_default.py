import torch
from utils import ROC, Autoenc, train_model, plot_loss, plot_det_map, calculate_det_map, plot_auc_and_epochloss
import os
import scipy.io as sio
import time

at = '_def'
results_name = 'name_of_results'

dataset_name = 'abu-airport-1'
print('Training on dataset:', dataset_name)

hidden_dim = 100
loss_func = 'matlab_loss' # matlab_loss, paper_loss, mse_loss
model = Autoenc(dataset_name, at, hidden_dim, dataset_folder='default', dataset_var="data")
loss_function = torch.nn.MSELoss()
model.learning_rate = 0.01
model.lambda_g = 0.01
model.epochs = 2000
model.batch_num = 100
model.batch_size = int(model.N/model.batch_num)

start_time = time.perf_counter()
model, epochs_loss, losses, auc_list = train_model(model, loss_func, es_bool=False)
end_time = time.perf_counter()
elapsed_time = end_time - start_time

epochs_loss_np = [loss.detach().numpy() for loss in epochs_loss] 

data_tensor = model.data_tensor

data_input = data_tensor[:,:].clone().transpose(1,2)
data_output, z_output = model(data_input)

det_map, new_X = calculate_det_map(data_tensor, data_output, model.m, model.n)
roc_auc_scores = ROC(det_map, model.y, 0)

print("Finished training on:", dataset_name)
print('Final AUC score:', dataset_name, 'is:', round(roc_auc_scores,4))
print('Total training time:', round(elapsed_time,2), '\n')

folder_name = 'results/' + dataset_name + '/'

if not os.path.exists(folder_name):
    os.makedirs(folder_name)

sio.savemat((folder_name + '/' + results_name + '.mat'), {'auc': roc_auc_scores, 'y': det_map, 'run_time': elapsed_time, 'epochs_loss': epochs_loss_np, 'list_of_auc': auc_list})

plot_auc_and_epochloss(auc_list, epochs_loss)
plot_det_map(new_X[:,:,0], model.y, det_map)
