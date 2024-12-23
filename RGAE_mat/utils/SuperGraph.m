function [SG,X_new,idex]=SuperGraph(data,S, mask, dataset_name)
% Construction of Laplacian matrix with SuperGraph
% INPUTS:
%   - data:  HSI data set (rows by columns by bands);
%   - S:     the number of superpixels.
% OUTPUT:
%   - SG:    the Laplacian matrix;
%   - X_new: the shuffled data set;
%   - idex:  the indices of the rows of 'X_new' corresponding to the reshaped 'data'.

    % Apply superpixel segmentation on compressed 'data'
    [M,N,L]=size(data);
    X=reshape(data,M*N,L);
    addpath(genpath('utils'))
    
    dim_reduction = 'pca'; 
    
    if dim_reduction == 'pca'
        disp('Dimentionality reduction using PCA.');
        Y=myPCA(data);
    elseif dim_reduction == 'clu'
        disp('Dimentionality reduction using Clustering.');
        Y=myClustering(data, mask);
    elseif dim_reduction == 'kpc'
        disp('Dimentionality reduction using KPCA.');
        % possible: 'linear', 'gauss', 'poly', 'sigm', 'exp', 'lapl'
        kpca_type = 'gauss';
        Y=myKPCA(data, kpca_type, dataset_name);
    elseif dim_reduction == 'ica'
        disp('Dimentionality reduction using ICA.');
    elseif dim_reduction == 'rpc'
%         load(join(['dim_red/RPCA/', dataset_name]));
%         Y = sparse_rank;
        Y = data;
        disp('Dimentionality reduction using RPCA.');
    elseif dim_reduction == 'pre'
        disp('Dimentionality reduction was done on forehand');
        Y = data;
    end  
    
%     Y=myPCA(data);
    y=Y(:,:,end);
%     image(y);
    y=(y-min(y(:)))./(max(y(:))-min(y(:)));
    
    
    
    Graph_separator = 'SLIC';
%     Graph_separator = 'QSHI';
%     Graph_separator = 'Felz';
%     Graph_separator = 'Wate';
    
    if Graph_separator == 'SLIC'
        
        % KPCA/C2 ---
%         dim_red_method = 'KPCA';
%         dim_red_img = join([dataset_name, '_KPCA.mat']);
%         
%         dim_red_method = 'C2';
%         dim_red_img = join([dataset_name, '_clustered.mat']);
%         
%         dim_red_img = join([dataset_name, '_clustered.mat']);
%         dim_red_path = join(['dim_red/', dim_red_method, ...
%             '/', dim_red_img]);
%         load(dim_red_path)
        
        if dim_reduction == 'rpc'
            y = sum(Y, 3);
        else
            y=Y(:,:,end);
        end
        y=(y-min(y(:)))./(max(y(:))-min(y(:)));
        % ---
        
        disp('Using SLIC image segmentation.');
        [labels,nums]=superpixels(y,S);
        
    elseif Graph_separator == 'QSHI'
        disp('Quickshift.');
%         maxdist=4;
%         ratio  = 1;
%         kernelsize = 2;
%         [quickshift_img, labels, map, gaps, E] = vl_quickseg(y, ratio, kernelsize, maxdist);
%         nums = max(max(labels));
        dim_red_method = 'KPCA';
        quickshift_img = join([dataset_name, '_qs.mat']);
        quickshift_path = join(['segmented/quickshift_', dim_red_method, ...
            '/', quickshift_img]);
        load(quickshift_path)
        
        y = double(y);
        y = y + 1; % Must add as it is made in python which is 0 indexed
        labels = y;
        nums = max(max(labels));
    elseif Graph_separator == 'Felz'
        disp('Felzenszwalb');
%         Felz_img = 'abu-airport-4_Felz';
%         Felz_img = 'abu-beach-4_Felz';
%         Felz_img = 'abu-urban-5_Felz';
%         file_path = 'graphs/felzenszwalb/';
%         file_path = 'graphs/C2/';
%         load(join([file_path, Felz_img]));
        
        
        dim_red_method = 'C2';
        felz_img = join([dataset_name, '_fz.mat']);
        felz_path = join(['segmented/felzenszwalb_', dim_red_method, ...
            '/', felz_img]);
        load(felz_path)
        
        
        y = double(y);
        y = y + 1; % Must add as it is made in python which is 0 indexed
        labels = y;
        nums = max(max(labels));
    elseif Graph_separator == 'Wate'
        disp('Watershed');
%         Felz_img = 'abu-airport-4_Felz';
%         Felz_img = 'abu-beach-4_Felz';
%         Felz_img = 'abu-urban-5_Felz';
%         file_path = 'graphs/felzenszwalb/';
%         file_path = 'graphs/C2/';
%         load(join([file_path, Felz_img]));
        
        
        dim_red_method = 'C2';
        water_img = join([dataset_name, '_ws.mat']);
        water_path = join(['segmented/watershed_', dim_red_method, ...
            '/', water_img]);
        load(water_path)
        
        
        y = double(y);
        y = y + 1; % Must add as it is made in python which is 0 indexed
        labels = y;
        nums = max(max(labels));
    end
    
    % Construct the Laplacian matrix with modified method
    W=sparse(M*N,M*N);
    spec2=4;X_new=[];
    idex=[];
    cnt=0;

    for num=1:nums
        idx=find(labels==num);
        K=size(idx,1);
        x=X(idx,:);
        X_new=[X_new;x];
        idex=[idex;idx];
        tmp=zeros(K); % BUGFIX
        for i=1:K
            s=x(i,:);
%             tmp=zeros(K); BUG - This should not be here
            for j=i+1:K
                tmp(i,j)=exp(-sum((s-x(j,:)).^2,2)/(2*spec2));
            end
        end
        W(cnt+1:cnt+K,cnt+1:cnt+K)=tmp+tmp';
        cnt=cnt+K;        
    end
    
    
%     
    SG=diag(sum(W))-W;
%     image(SG);
%     disp(max(W));
%     disp(min(min(W)));
end

