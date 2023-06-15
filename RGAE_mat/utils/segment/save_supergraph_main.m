addpath(genpath('utils'))

clear all;
close all;
clc
warning('off');

model_name = 'def';

file_path = 'datasets/';
load('parameters/default');

for i = 1:length(dataset_list(:,1))

    % Find correct end idx for dataset_name
    for j = 1:length(dataset_list(i,:))
        if dataset_list(i,j) == '.'
            idx = j - 1;
        end
    end
    
    dataset_name = dataset_list(i,1:idx);
    disp(dataset_name);
    S = S_list(i);
    
    load(join([file_path, dataset_name]));
    data = (data-min(data(:)))./(max(data(:))-min(data(:)));
    
    [SG,X_new,idex] = SuperGraph_save(data,S);
    idex = idex-1;
    
    folder_path = join(['segmented/graphs/', dataset_name]);
    
    % Check if the folder exists
    if ~exist(folder_path, 'dir')
        % If the folder does not exist, create it using the mkdir function
        mkdir(folder_path);
    end

    filename = join([folder_path, '/', dataset_name, '_', model_name, '.mat']);
    save(filename);

end