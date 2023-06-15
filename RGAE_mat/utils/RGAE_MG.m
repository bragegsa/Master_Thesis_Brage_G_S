function y = RGAE_MG(data_1, data_2,lambda,S,n_hid, map, epochs, lr, dataset_name)
% Methodology.
% INPUTS:
%   - data:   HSI data set (rows by columns by bands);
%   - lambda: the tradeoff parameter;
%   - S:      the number of superpixels;
%   - n_hid:  the number of hidden layer nodes.
% OUTPUT:
%   - y:     final detection map (rows by columns).
    
    % Laplacian matrix construction with SuperGraph
    [SG_1,X_new,idex_1]=SuperGraph_MK(data_1,S,map,dataset_name);
    [SG_2,X_new,idex_2]=SuperGraph_MK(data_2,S,map,dataset_name);

    % RGAE with ADAM
    tic;
    y_tmp=myRGAE_MG(X_new,SG_1,SG_2,lambda,n_hid,map,epochs,lr,idex_2);
    toc;
    
    % Output
    zips=[idex_2,y_tmp']; % Recover the original image with 'idex'
    
    zips_sort=sortrows(zips,1);
    y=zips_sort(:,2);
end
