addpath(genpath('utils'))

clear all;
close all;
clc
warning('off');

folder_path = 'datasets';  % replace with the path to your folder
file_list = dir(fullfile(folder_path, '*.mat'));  % replace '*.txt' with the extension of the files you want to iterate over

for i = 1:length(file_list)
    filename = file_list(i).name;
    fullpath = fullfile(folder_path, filename);
    % Do something with the file here
    load(join(fullpath));
    disp(fullpath);
    
    data = (data-min(data(:)))./(max(data(:))-min(data(:)));

    pca = myPCA(data);
    pca = (pca-min(pca(:)))./(max(pca(:))-min(pca(:)));

    
    save_path = 'dim_red/PCA/';
    full_save_path = join([save_path, filename]);
    save(full_save_path, 'pca');
end

% Y=myPCA(data)