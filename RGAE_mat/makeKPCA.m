addpath(genpath('utils'))

clear all;
close all;
clc
warning('off');

% datasets = {'abu-beach-1', 'abu-beach-4'};
datasets = {'abu-airport-1', 'abu-airport-2', 'abu-airport-3', 'abu-airport-4', ...
    'abu-beach-2', 'abu-beach-3', ...
    'abu-urban-1', 'abu-urban-2', 'abu-urban-3', 'abu-urban-4', 'abu-urban-5'};
kpca_kernels = {'lapl', 'poly', 'sigm'};
% kpca_kernels = {'lapl','sigm'};
dims = 300;

for i = 1:length(datasets)
    for j = 1:length(kpca_kernels)
        
        dataset_name = datasets{i};
        kpca_kernel = kpca_kernels{j};
        
        disp_string = join(['KPCA on ', dataset_name, ' using ', ...
            kpca_kernel , '.']);
        disp(disp_string);

        % Loading dataset
        file_path = 'datasets/';
        load(join([file_path, dataset_name]));

        % Normalizing the data
        data = (data-min(data(:)))./(max(data(:))-min(data(:)));
        
        % Calculating KPCA
        tic;
        Y=myKPCA(data, kpca_kernel, dataset_name, dims);
        time = toc;
        
        filename = join(['dim_red/KPCA_time/', dataset_name, '_', kpca_kernel, '_', int2str(dims)]);
        save(filename, 'time');
    end
end