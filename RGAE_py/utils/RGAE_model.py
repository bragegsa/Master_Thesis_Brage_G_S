import torch
import utils
from utils import load_dataset, load_graph, RGAE_mat_loss, early_stopping, early_stopping_2, RGAE_paper_loss, RGAE_MSE_loss
from torchvision import transforms
from sklearn.preprocessing import MinMaxScaler
import numpy as np

class Autoenc(torch.nn.Module):
    def __init__(self, dataset_name, at, hidden_dim, dataset_folder='default', dataset_var='data'):
        super().__init__()

        # Get dataset
        X, self.y = load_dataset(dataset_name, dataset_var, dataset_folder)
        self.X = np.abs(X)
        [self.m,self.n,self.L] = np.shape(self.X)
        X_2D = self.X.reshape((self.m*self.n, self.L))
        self.N = self.m*self.n

        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        self.SG, X_sg, idx = load_graph(dataset_name, at)

        X_2D = X_2D.astype('float32')
        scaler = MinMaxScaler()
        X_2D = scaler.fit_transform(X_2D)
        self.data_tensor = transform(X_2D)
        self.X_2D = X_2D

        self.lambda_g = 0.01
        self.learning_rate = 0.01

        # Define network parameters
        self.batch_num = 10
        self.batch_size = int(self.N/self.batch_num)
        self.epochs = 400
        self.outputs = []
        self.losses = []
        self.hidden_dim = hidden_dim

        
        # Define the layers of the autoencoder
        self.W1 = torch.nn.Parameter(0.01*torch.rand(self.hidden_dim, self.L))
        self.b1 = torch.nn.Parameter(torch.rand(self.hidden_dim))
        self.W2 = torch.nn.Parameter(0.01*torch.rand(self.L, self.hidden_dim))
        self.b2 = torch.nn.Parameter(torch.rand(self.L))


    def forward(self, x):
        ph, n_rows, n_cols = x.size()

        # Encoder
        z = torch.sigmoid(torch.matmul(self.W1, x) + self.b1.repeat(n_cols, 1).transpose(0, 1))
        # print('z shape', z.shape)
        
        # Decoder
        x_hat = torch.sigmoid(torch.matmul(self.W2, z) + self.b2.repeat(n_cols, 1).transpose(0,1))
        return x_hat, z
    
def train_model(model, loss_func='matlab_loss', es_bool=True):

    epochs = model.epochs
    outputs = []
    losses = []
    batch_num = model.batch_num
    batch_size = model.batch_size
    data_tensor = model.data_tensor

    L_matrix = model.SG
    lambda_g = model.lambda_g
    learning_rate = model.learning_rate

    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate) 
    epoch_loss = 0
    epochs_loss = []
    auc_list = []

    for epoch in range(epochs):

        print('Epoch:', epoch)
        ind = np.random.permutation(model.N)
        epoch_loss = 0

        for i in range(batch_num):

            x = data_tensor[:,ind[i*batch_size:(i+1)*batch_size]].transpose(1, 2)
            sg = L_matrix[ind[i*batch_size:(i+1)*batch_size],:]
            sg = sg[:,ind[i*batch_size:(i+1)*batch_size]].toarray()

            reconstructed, z = model(x)
            
            if loss_func == 'matlab_loss':
                loss = RGAE_mat_loss(reconstructed[0,:,:], x[0,:,:], z[0,:,:],lambda_g,sg,model.L)
            elif loss_func == 'paper_loss':
                loss = RGAE_paper_loss(reconstructed[0,:,:], x[0,:,:], z[0,:,:],lambda_g,sg,batch_size,model.L)
            elif loss_func == 'mse_loss':
                loss = RGAE_MSE_loss(reconstructed[0,:,:], x[0,:,:], z[0,:,:],lambda_g,sg)

            epoch_loss += loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss)

        track_auc = True
        if track_auc == True:
            auc = utils.get_auc_training(model, data_tensor, model.y)
            auc_list.append(round(auc, 4))
            print('Previous auc:', auc)
        
        epochs_loss.append(epoch_loss)
        print('Previous loss:', epoch_loss.detach().numpy())
        

        if(early_stopping_2(epochs_loss) and es_bool):
            break
    
    outputs.append((epochs, x, reconstructed))

    return model, epochs_loss, losses, auc_list