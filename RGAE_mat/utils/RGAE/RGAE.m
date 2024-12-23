function y = RGAE(data,lambda,S,n_hid, map, epochs, lr, dataset_name, network_setup)
% Methodology.
% INPUTS:
%   - data:   HSI data set (rows by columns by bands);
%   - lambda: the tradeoff parameter;
%   - S:      the number of superpixels;
%   - n_hid:  the number of hidden layer nodes.
% OUTPUT:
%   - y:     final detection map (rows by columns).
    
    % Laplacian matrix construction with SuperGraph
    [SG,X_new,idex]=SuperGraph(data,S, map, dataset_name);
    
    % Training RGAE
%     y_tmp=myRGAE(X_new,SG,lambda,n_hid, map, epochs, lr, idex);

    % RGAE with momentum
%     y_tmp=myRGAE_momentum(X_new,SG,lambda,n_hid, map, epochs, lr, idex);

    % RGAE with RMSP
%     y_tmp=myRGAE_RMSP(X_new,SG,lambda,n_hid, map, epochs, lr, idex);

    % RGAE with ADAM
    y_tmp=myRGAE_ADAM(X_new,SG,lambda,n_hid, map, epochs, lr, idex, dataset_name, network_setup);

    % RGAE with multiple layers in encoder and decoder
%     y_tmp=myRGAE_multi(X_new,SG,lambda,n_hid, map, epochs, lr, idex);
    
    % RGAE with multiple layers in encoder and decoder
%     y_tmp=myRGAE_ADAM_multi(X_new,SG,lambda,n_hid, map, epochs, lr, idex);

    % RGAE test
%     y_tmp=myRGAE_test(X_new,SG,lambda,n_hid, map, epochs, lr, idex, network_setup, dataset_name);
    
    % Output
    zips=[idex,y_tmp'];            % Recover the original image with 'idex'
    
    zips_sort=sortrows(zips,1);
    y=zips_sort(:,2);
    disp('stop');
end
