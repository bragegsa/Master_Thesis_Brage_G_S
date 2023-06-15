
function [y,time] = RPCA_fuse(data,rpca_low_rank,lambda,S,n_hid,map,epochs,lr,dataset_name,network_setup)
    % Methodology.
    % INPUTS:
    %   - data:   HSI data set (rows by columns by bands);
    %   - lambda: the tradeoff parameter;
    %   - S:      the number of superpixels;
    %   - n_hid:  the number of hidden layer nodes.
    % OUTPUT:
    %   - best_y:     final detection map (rows by columns).
    
    network_setup_data = join([network_setup, '_rpca_data']);
    network_setup_low_rank = join([network_setup, '_rpca_low_rank']);
    
    tic;
    y_data = RGAE(data,lambda,S,n_hid, map, epochs, lr, ...
        dataset_name, network_setup_data);
    y_low_rank = RGAE(rpca_low_rank,lambda,S,n_hid, map, epochs, lr, ...
        dataset_name, network_setup_low_rank);
    time = toc;
    
    y_data=reshape(y_data,size(map,1),size(map,2));
    y_low_rank=reshape(y_low_rank,size(map,1),size(map,2));
    
%     alpha_values = 0:0.01:1;
%     best_alpha = 0;
%     best_auc = 0;
%     best_y;
% 
%     for alpha=alpha_values
%         y = alpha*y_sparse + (1-alpha)*y_low_rank;
%         AUC = ROC(y, map, 0);
%         if AUC > best_auc
%             best_auc = AUC;
%             best_alpha = alpha;
%             best_y = y;
%         end
%     end    

    y = y_data - y_low_rank;
end