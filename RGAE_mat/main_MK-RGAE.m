addpath(genpath('utils'))

clear all;
close all;
clc
warning('off');

% Her bør KPCA hentes først, ikke ferdig data
% dataset_name = 'abu-airport-2';
dataset_name = 'abu-beach-4';
% dataset_name = 'abu-urban-4';
file_path = 'datasets/';
load(join([file_path, dataset_name]));

mask = map;

% file_path = 'dim_red/KPCA3D/';
file_path = 'dim_red/KPCA/sigm/';
kpca_type_1 = 'sigm';
kpca_type_2 = 'lapl';
% space = '_';
space = '';
% max_dim = 30;
max_dim = 100;
% max_dim = 300;

% Parameters ---
lambda = 1e-4; % MK not 0 or 1 if lambda = 0?
% lambda = 0;
S=150;
n_hid=100;

lr = 0.01;
epochs = 10;
% ---

file_name = [file_path, dataset_name, space, kpca_type_1];
load(join([file_name]));
% Data_KPCA_3D = abu(:,:,1:max_dim);
Data_KPCA_3D = Y(:,:,1:max_dim);
data1 = real(Data_KPCA_3D);
data1 = (data1-min(data1(:)))./(max(data1(:))-min(data1(:)));
% figure, imshow(data1(:,:,1))

[h_y1, w_y1, l_y1] = size(Y);

file_path = 'dim_red/KPCA/lapl/';
file_name = [file_path, dataset_name, space, kpca_type_2];
load(join([file_name]));
% Data_KPCA_3D = abu(:,:,1:max_dim);
Data_KPCA_3D = Y(:,:,1:max_dim);
data2 = real(Data_KPCA_3D);
data2 = (data2-min(data2(:)))./(max(data2(:))-min(data2(:)));
% figure, imshow(data2(:,:,1))

[h_y2, w_y2, l_y2] = size(Y);


tic;
y1 = RGAE(data1,lambda,S,n_hid, map, epochs, lr, dataset_name, 'test');
% y1 = 1 - y1; % If laplace, exp
time = toc;

tic;
y2 = RGAE(data2,lambda,S,n_hid, map, epochs, lr, dataset_name, 'test');
y2 = 1 - y2; % If laplace, exp - Abu-beach-2 needs not to be inversed
time = toc;

AUC1=ROC(y1,map,0);
disp(AUC1);
AUC2=ROC(y2,map,0);
disp(AUC2);


y1=reshape(y1,h_y1,w_y1);
y2=reshape(y2,h_y2,w_y2);
% figure, imshow(y1);
% figure, imshow(y2);

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


filename = join(['results/', dataset_name, '/', 'MKPCA_', int2str(max_dim)]);
save(filename);
