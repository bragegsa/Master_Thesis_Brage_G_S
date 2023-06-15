function y = RGAE_lowrank(data,low_rank,lambda,S,n_hid, map, epochs, lr, dataset_name, network_setup)
% Methodology.
% INPUTS:
%   - data:   HSI data set (rows by columns by bands);
%   - lambda: the tradeoff parameter;
%   - S:      the number of superpixels;
%   - n_hid:  the number of hidden layer nodes.
% OUTPUT:
%   - y:     final detection map (rows by columns).
    
    % Laplacian matrix construction with SuperGraph
    [SG,X_new,idex,X_data]=SuperGraph_split(data,low_rank,S, map, dataset_name);

    % RGAE with ADAM
    y_tmp=myRGAE_lowrank(X_new,X_data,SG,lambda,n_hid, map, epochs, lr, idex, dataset_name, network_setup);

    % Output
    zips=[idex,y_tmp'];            % Recover the original image with 'idex'
    
    zips_sort=sortrows(zips,1);
    y=zips_sort(:,2);
    disp('stop');
end
