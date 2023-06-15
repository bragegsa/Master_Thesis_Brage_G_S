import torch
import utils
from utils import load_dataset, load_graph, interpolation_loss, early_stopping_2, get_auc_training_int, interpolation_loss2
from torchvision import transforms
from sklearn.preprocessing import MinMaxScaler
import numpy as np

class RGAE_int(torch.nn.Module):
    def __init__(self, dataset_name, at, hidden_dim, dataset_folder='default', dataset_var='data', dims=0):
        super().__init__()

        # Get defaul dataset
        X2, y2 = load_dataset(dataset_name, 'data', 'default')
        self.X2 = np.abs(X2[:,:,:])
        [self.m,self.n,self.L] = np.shape(self.X2)

        # Get dataset
        X1, self.y = load_dataset(dataset_name, dataset_var, dataset_folder)
        self.X1 = np.abs(X1[:,:,0:self.L])
        

        print('X1 shape', self.X1.shape)
        print('X2 shape', self.X2.shape)

        
        X1_2D = self.X1.reshape((self.m*self.n, self.L))
        X2_2D = self.X2.reshape((self.m*self.n, self.L))
        self.N = self.m*self.n

        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        self.SG, X_sg, idx = load_graph(dataset_name, at)

        X1_2D = X1_2D.astype('float32')
        scaler = MinMaxScaler()
        X1_2D = scaler.fit_transform(X1_2D)
        self.data_tensor1 = transform(X1_2D)
        self.X1_2D = X1_2D

        X2_2D = X2_2D.astype('float32')
        scaler = MinMaxScaler()
        X2_2D = scaler.fit_transform(X2_2D)
        self.data_tensor2 = transform(X2_2D)
        self.X2_2D = X2_2D

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
        self.We1 = torch.nn.Parameter(0.01*torch.rand(self.hidden_dim, self.L))
        self.be1 = torch.nn.Parameter(torch.rand(self.hidden_dim))
        # self.We2 = torch.nn.Parameter(0.01*torch.rand(self.hidden_dim, self.L))
        # self.be2 = torch.nn.Parameter(torch.rand(self.hidden_dim))
        self.W2 = torch.nn.Parameter(0.01*torch.rand(self.L, self.hidden_dim))
        self.b2 = torch.nn.Parameter(torch.rand(self.L))

        self.alpha = 0

    def forward(self, x1, x2):
        ph, n_rows, n_cols = x1.size()
        
        # Encoder
        z1 = torch.sigmoid(torch.matmul(self.We1, x1) + self.be1.repeat(n_cols, 1).transpose(0, 1))
        z2 = torch.sigmoid(torch.matmul(self.We1, x2) + self.be1.repeat(n_cols, 1).transpose(0, 1))

        z = self.alpha * z1 + (1 - self.alpha) * z2
        
        # Decoder
        x_hat = torch.sigmoid(torch.matmul(self.W2, z) + self.b2.repeat(n_cols, 1).transpose(0,1))
        return x_hat, z
    
def train_RGAE_int(model, es_bool=True):

    epochs = model.epochs
    outputs = []
    losses = []
    batch_num = model.batch_num
    batch_size = model.batch_size

    data_tensor1 = model.data_tensor1
    data_tensor2 = model.data_tensor2

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
        ind2 = np.random.permutation(model.N)
        epoch_loss = 0

        for i in range(batch_num):

            x1 = data_tensor1[:,ind[i*batch_size:(i+1)*batch_size]].transpose(1, 2)
            x2 = data_tensor2[:,ind2[i*batch_size:(i+1)*batch_size]].transpose(1, 2)
            sg1 = L_matrix[ind[i*batch_size:(i+1)*batch_size],:]
            sg1 = sg1[:,ind[i*batch_size:(i+1)*batch_size]].toarray()

            sg2 = L_matrix[ind2[i*batch_size:(i+1)*batch_size],:]
            sg2 = sg2[:,ind2[i*batch_size:(i+1)*batch_size]].toarray()

            reconstructed, z = model(x1, x2)

            loss = interpolation_loss2(reconstructed[0,:,:], x1[0,:,:], x2[0,:,:], z[0,:,:], lambda_g, sg1, sg2, model.alpha)

            epoch_loss += loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss)

        track_auc = True
        if track_auc == True:
            auc = utils.get_auc_training_int(model, data_tensor1, data_tensor2, model.y, model.alpha)
            auc_list.append(round(auc, 4))
            print('Previous auc:', auc)
        
        epochs_loss.append(epoch_loss)
        print('Previous loss:', epoch_loss.detach().numpy())
        

        # if(early_stopping_2(epochs_loss) and es_bool and auc > 0.8):
        #     break
        if(early_stopping_2(epochs_loss) and es_bool):
            break
    
    outputs.append((epochs, x1, x2, reconstructed))

    return model, epochs_loss, losses, auc_list