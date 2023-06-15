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

input = 'mk_____';
% kernel = 'PCA';
kernel_size = 300;
model_name = 'multikernel_RGAE_100_sigm_lapl';
network_setup = 'sigsig';

file_path = 'datasets/';
% file_path = 'dim_red/RPCA/';
load('parameters/default');

% Parameters to optimize: (default values)
n_hid=50;
epochs = 500;

AUC_scores = [];


for i = 1:length(dataset_list(:,1))

    % Find correct end idx for dataset_name
    for j = 1:length(dataset_list(i,:))
        if dataset_list(i,j) == '.'
            idx = j - 1;
        end
    end
    
    dataset_name = dataset_list(i,1:idx);
    
    lr = lr_list(i);
    S = S_list(i);
    lambda = lambda_list(i);
    
    disp(join(['Training on dataset: ', dataset_name]));
    load(join([file_path, dataset_name]));
    
    mask = map;
    
    if input == 'default'
        disp("Using default input.");
        
        data = (data-min(data(:)))./(max(data(:))-min(data(:)));
        
        tic;
        y = RGAE(data,lambda,S,n_hid, map, epochs, lr, dataset_name, network_setup);
        time = toc;
        y = reshape(y,size(map,1),size(map,2));
        
        
    elseif input == 'mk_____'
        max_dim = 100;
        space = '';
        kpca_type_1 = 'sigm';
        kpca_type_2 = 'lapl';
        
        file_path_kpca = 'dim_red/KPCA/';
        file_name = [file_path_kpca, kpca_type_1, '/', dataset_name, kpca_type_1];
        load(join(file_name));
        
        data1 = real(abu(:,:,1:max_dim));
        data1 = (data1-min(data1(:)))./(max(data1(:))-min(data1(:)));
        
        [h, w, l] = size(data1);
        
        file_name = [file_path_kpca, kpca_type_2, '/', dataset_name, kpca_type_2];
        load(join(file_name));
        
        data2 = real(abu(:,:,1:max_dim));
        data2 = (data2-min(data2(:)))./(max(data2(:))-min(data2(:)));
        
        tic;
        y1 = RGAE(data1,lambda,S,n_hid, map, epochs, lr, dataset_name, network_setup);
        y2 = RGAE(data2,lambda,S,n_hid, map, epochs, lr, dataset_name, network_setup);
        time = toc;
        
        AUC1=ROC(y1,map,0);
        disp(AUC1);
        AUC2=ROC(y2,map,0);
        disp(AUC2);
        
        y1=reshape(y1,h, w);
        y2=reshape(y2,h, w);
        
        alpha_values = 0:0.01:1;
        best_alpha = 0;
        best_auc = 0;

        for alpha=alpha_values
            y = alpha*y1 + (1-alpha)*y2;
            AUC = ROC(y, map, 0);
            if AUC > best_auc
                best_auc = AUC;
                best_alpha = alpha;
            end
        end

        print_statement = ['The best alpha value was ', num2str(best_alpha), ...
            ' with an AUC score of ', num2str(best_auc), '.'];

        disp(join(print_statement));
        
    elseif input == 'multi__'
        
        disp("Using default input. Multi-RGAE");
        
        data = (data-min(data(:)))./(max(data(:))-min(data(:)));
        
        tic;
        y = RGAE(data,lambda,S,n_hid, map, epochs, lr, dataset_name, network_setup);
        time = toc;
        y = reshape(y,size(map,1),size(map,2));
    
    elseif input == 'PCA____'
        disp("Using PCA input.");
%         kernel = 'sigm';
        file_path_kpca = join(['dim_red/PCA/',dataset_name]);
        
        load(file_path_kpca);
%         data = abs(pca(:,:,1:kernel_size));
        data = pca;
        data = (data-min(data(:)))./(max(data(:))-min(data(:)));
        
        tic;
        y = RGAE(data,lambda,S,n_hid, map, epochs, lr, dataset_name, network_setup);
        time = toc;
        y = reshape(y,size(map,1),size(map,2));    
    elseif input == 'KPCA___'
        disp("Using KPCA input.");
%         kernel = 'sigm';
        file_path_kpca = join(['dim_red/KPCA/', kernel, '/', dataset_name, kernel]);
        
        load(file_path_kpca);
        data = abs(abu(:,:,1:kernel_size));
        data = (data-min(data(:)))./(max(data(:))-min(data(:)));
        
        tic;
        y = RGAE(data,lambda,S,n_hid, map, epochs, lr, dataset_name, network_setup);
        time = toc;
        y = reshape(y,size(map,1),size(map,2));
    
    elseif input == 'rpcaSae'
        
        disp('Using signal space as input');
        
        data = (data-min(data(:)))./(max(data(:))-min(data(:)));
        
        file_path_rpca = 'dim_red/RPCA/';
        load(join([file_path_rpca, dataset_name]));
        rpca_low_rank = sparse_rank;
        rpca_low_rank = (rpca_low_rank-min(rpca_low_rank(:)))./(max(rpca_low_rank(:))-min(rpca_low_rank(:)));
    
        tic;
        y = RGAE(rpca_low_rank,lambda,S,n_hid, map, epochs, lr, dataset_name, network_setup);
%         y = RGAE_lowrank(data,rpca_low_rank,lambda,S,n_hid, map, epochs, lr, dataset_name, network_setup);
        time = toc;
        y = reshape(y,size(map,1),size(map,2));
        
    elseif input == 'rpca_si'
        
        disp('Using signal space for graph creation and default HSI as input');
        
        data = (data-min(data(:)))./(max(data(:))-min(data(:)));
        
        file_path_rpca = 'dim_red/RPCA/';
        load(join([file_path_rpca, dataset_name]));
        rpca_low_rank = low_rank;
        rpca_low_rank = (rpca_low_rank-min(rpca_low_rank(:)))./(max(rpca_low_rank(:))-min(rpca_low_rank(:)));
        
        tic;
        y = RGAE_split(rpca_low_rank,data,lambda,S,n_hid, map, epochs, lr, dataset_name, network_setup);
        time = toc;
        y = reshape(y,size(map,1),size(map,2));
        
    elseif input == 'rpca_sn'

        disp('Using signal space for graph creation and noise as input');
        
        file_path_rpca = 'dim_red/RPCA/';
        load(join([file_path_rpca, dataset_name]));
        
        data = sparse_rank;
        data = (data-min(data(:)))./(max(data(:))-min(data(:)));
        
        rpca_low_rank = low_rank;
        rpca_low_rank = (rpca_low_rank-min(rpca_low_rank(:)))./(max(rpca_low_rank(:))-min(rpca_low_rank(:)));
        
        tic;
        y = RGAE_split(rpca_low_rank,data,lambda,S,n_hid, map, epochs, lr, dataset_name, network_setup);
        time = toc;
        y = reshape(y,size(map,1),size(map,2));
        
    elseif input == 'rpca_sp'
        disp("Using RPCA sparse as input.");
        
        file_path_rpca = 'dim_red/RPCA/';
        load(join([file_path_rpca, dataset_name]));
        data = sparse_rank;
        
        data = (data-min(data(:)))./(max(data(:))-min(data(:)));
    
        tic;
        y = RGAE(data,lambda,S,n_hid, map, epochs, lr, dataset_name, network_setup);
        time = toc;
        y = reshape(y,size(map,1),size(map,2));
        
    elseif input == 'rpca_dc'
        disp("Using RPCA decision fusion.");
        
        file_path_rpca = 'dim_red/RPCA/';
        load(join([file_path_rpca, dataset_name]));
        data = (data-min(data(:)))./(max(data(:))-min(data(:)));
%         rpca_sparse = sparse_rank;
        rpca_low_rank = low_rank;
%         rpca_sparse = (rpca_sparse-min(rpca_sparse(:)))./(max(rpca_sparse(:))-min(rpca_sparse(:)));
        rpca_low_rank = (rpca_low_rank-min(rpca_low_rank(:)))./(max(rpca_low_rank(:))-min(rpca_low_rank(:)));
        
        [y,time] = RPCA_fuse(data,rpca_low_rank,lambda,...
            S,n_hid,map,epochs,lr,dataset_name,network_setup);
        
    elseif input == 'lowrank'
        disp("Using lowrank.");
        
        file_path_rpca = 'dim_red/RPCA/';
        load(join([file_path_rpca, dataset_name]));
        data = (data-min(data(:)))./(max(data(:))-min(data(:)));
        rpca_sparse = sparse_rank;
        rpca_low_rank = low_rank;
%         [M,N,L]=size(rpca_sparse);
        
        rpca_sparse = (rpca_sparse-min(rpca_sparse(:)))./(max(rpca_sparse(:))-min(rpca_sparse(:)));
        rpca_low_rank = (rpca_low_rank-min(rpca_low_rank(:)))./(max(rpca_low_rank(:))-min(rpca_low_rank(:)));
        
        tic;
        y_lr = RGAE(rpca_low_rank,lambda,...
            S,n_hid,map,epochs,lr,dataset_name,network_setup);
        time = toc;
        
        y_lr = reshape(y_lr, size(map,1),size(map,2));
%         y_sparse =reshape(rpca_sparse,M*N,L);
        y = rpca_sparse(:,:,1) - y_lr;
        
    elseif input == 'rpca_lr'
        disp("Using default input, with low rank rpca subtraction after training.");
        
        data = (data-min(data(:)))./(max(data(:))-min(data(:)));
        
        file_path_rpca = 'dim_red/RPCA/';
        load(join([file_path_rpca, dataset_name]));
        rpca_low_rank = low_rank;
        rpca_low_rank = (rpca_low_rank-min(rpca_low_rank(:)))./(max(rpca_low_rank(:))-min(rpca_low_rank(:)));
        
        tic;
        y = RGAE(data,lambda,S,n_hid, map, epochs, lr, dataset_name, network_setup);
        time = toc;
        y = reshape(y,size(map,1),size(map,2));
        y = y - rpca_low_rank(:,:,1);
        y = (y-min(y(:)))./(max(y(:))-min(y(:)));
        
    end
    
    AUC=ROC(y,map,0);   
     
    disp(join(['Finished training with AUC score of: ', num2str(AUC)]));
    
    % Specify the folder path
    folder_path = join(['results_master/', dataset_name]);

    % Check if the folder exists
    if ~exist(folder_path, 'dir')
        % If the folder does not exist, create it using the mkdir function
        mkdir(folder_path);
    end

    filename = join([folder_path, '/', dataset_name, '_', model_name, '.mat']);
    save(filename);
    
    AUC_scores(end + 1) = AUC;
end

str_list = strjoin(string(AUC_scores), ',');
disp(str_list);