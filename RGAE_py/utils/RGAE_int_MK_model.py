import torch
import utils
from utils import load_dataset, load_graph, interpolation_loss, early_stopping_2, get_auc_training_int, interpolation_loss2
from torchvision import transforms
from sklearn.preprocessing import MinMaxScaler
import numpy as np

class RGAE_int_MK(torch.nn.Module):
    def __init__(self, dataset_name, at, hidden_dim):
        super().__init__()

        # Get defaul dataset
        X2, y2 = load_dataset(dataset_name, 'abu', 'KPCA/sigm')
        self.X2 = np.abs(X2[:,:,:])
        [self.m,self.n,self.L] = np.shape(self.X2)

        # Get dataset
        X1, self.y = load_dataset(dataset_name, 'abu', 'KPCA/lapl')
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