function Y=myKPCA(data, kpca_type, dataset_name, dims)
% KPCA
    
    DataTest =data;
    
    [num_rows, num_cols, N] = size(DataTest);  
    M = num_rows * num_cols;

    for i=1:N
        DataTest(:,:,i) = (DataTest(:,:,i)-min(min(DataTest(:,:,i)))) / (max(max(DataTest(:,:,i))-min(min(DataTest(:,:,i)))));
    end

    Data = reshape(DataTest,M,N); 

    tic;
    
%     dims = 300;
    
    % set kernel function, possible: 'linear', 'gauss', 'poly', 'sigm'
    % 'exp', 'lapl'
    kernel = Kernel('type', kpca_type, 'width', 2); % 0.5 --> 2
    % parameter setting
    parameter = struct('application', 'dr', 'dim', dims, 'kernel', kernel); % 2 --> N  300
    % build a KPCA object
    kpca = KernelPCA(parameter);
    % train KPCA model using given data
    X_map = kpca.train(Data);

    toc;
  
    Data_KPCA_3D = reshape(X_map,num_rows, num_cols, dims);
    
%     imshow(Data_KPCA_3D(:,:,1));
    
    Y = Data_KPCA_3D(:,:,:);
    
end
