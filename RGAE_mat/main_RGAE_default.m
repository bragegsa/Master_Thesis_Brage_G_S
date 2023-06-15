%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% APPLICATION:
%   Hyperspectral Anomaly Detection.
% INPUTS:
%   - data:   HSI data set (rows by columns by bands);
%   - lambda: the tradeoff parameter;
%   - S:      the number of superpixels;
%   - n_hid:  the number of hidden layer nodes.
% OUTPUTS:
%   - y:    final detection map (rows by columns);
%   - AUC:  AUC value of 'y'.
%  REFERENCE:
%   G. Fan, Y. Ma, X. Mei, F. Fan, J. Huang and J. Ma, "Hyperspectral Anomaly
%   Detection With Robust Graph Autoencoders," IEEE Transactions on Geoscience 
%   and Remote Sensing, 2021.
%   G. Fan, Y. Ma, J. Huang, X. Mei and J. Ma, "Robust Graph Autoencoder for 
%   Hyperspectral Anomaly Detection," ICASSP 2021 - 2021 IEEE International 
%   Conference on Acoustics, Speech and Signal Processing (ICASSP), 2021, 
%   pp. 1830-1834.
% Written and sorted by Ganghui Fan in 2021. All rights reserved.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
addpath(genpath('utils'))

clear all;
close all;
clc
warning('off');

model_name = 'default';
network_setup = 'test';
% network_setup = 'sigsig';

% Loading the dataset
dataset_name = 'abu-airport-2';
% dataset_name = 'abu-beach-3';
% dataset_name = 'abu-urban-1';
file_path = 'datasets/';
load(join([file_path, dataset_name]));

% file_path = 'dim_red/SAVED_KPCA/';
% load(join([file_path, dataset_name, 'poly']));

% data = real(abu);

mask = map;

% Normalizing the data
data = (data-min(data(:)))./(max(data(:))-min(data(:)));
data = data(:,:,1:50);

% Parameters to optimize: (default values)
lambda = 0.0001;
S=50;
n_hid=100;

lr = 0.01;
epochs = 300;

tic;
y = RGAE(data,lambda,S,n_hid, map, epochs, lr, dataset_name, network_setup);
time = toc;

y = reshape(y,size(map,1),size(map,2));

AUC=ROC(y,map,0);
disp(AUC);

% Specify the folder path
folder_path = join(['results_master/', dataset_name]);

% Check if the folder exists
if ~exist(folder_path, 'dir')
    % If the folder does not exist, create it using the mkdir function
    mkdir(folder_path);
end

filename = join([folder_path, '/', dataset_name, '_', network_setup, '_', int2str(n_hid)]);
save(filename);

