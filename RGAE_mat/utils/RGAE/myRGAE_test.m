function y=myRGAE_test(X,SG,lamda,hid, map, epochs, lr, idex, network_setup, dataset_name)%
% Training of RGAE for hyperspectral anomaly detection
% INPUTS:
%   - X:     HSI data set (rows*columns by bands);
%   - SG:    the Laplacian matrix;
%   - lambda:the tradeoff parameter;
%   - hid:   the number of hidden layer nodes.
%   - epochs:  the number of epochs
%   - lr:      the learning rate
%   - map:     the ground truth
%   - idex:  the indices of the rows of 'X_new' corresponding to the reshaped 'data'.
% OUTPUT:
%   - y:     final detection map (rows by columns).

    % Parameter settings
    [n,L]=size(X);X=X';
    
%     network_setup = 'defsig';
%     network_setup = 'sigsig';
%     network_setup = 'relsig';
    
    batch_num=10;batch_size=n/batch_num;    % training with MBGD
    
    We1=0.01*rand(hid,L);be1=rand(hid,1);     % weights and biases for encoder
    m_e1 = 0; v_e1 = 0;
    m_e1b = 0; v_e1b = 0;
    
    x_size = We1*X(:,batch_size+1:2*batch_size); % Size of output after first encoding layer
    x_r = size(x_size, 1);
    hid_2 = ceil(hid/2);
%     hid_2 = hid;
    
    We2=0.01*rand(hid_2,x_r);be2=rand(hid_2,1);     % weights and biases for encoder
    m_e2 = 0; v_e2 = 0;
    m_e2b = 0; v_e2b = 0;
    
    Wd1=0.01*rand(x_r,hid_2);bd1=rand(x_r,1);       % weights and biases for decoder
    m_d1 = 0; v_d1 = 0;
    m_d1b = 0; v_d1b = 0;
    Wd2=0.01*rand(L,hid);bd2=rand(L,1);       % weights and biases for decoder    
    m_d2 = 0; v_d2 = 0;
    m_d2b = 0; v_d2b = 0;
    
    t = 0;
    
    n_AUC = 10;
    AUC_values = zeros(1,n_AUC); % Holds the previous n_AUC AUC values for dropout
    All_AUC = [];
    error_curve = [];
    epoch_loss_curve = [];
    
    for epoch=1:epochs
        ind=randperm(n);
        
        epoch_loss = 0;
        for j=1:batch_num
            x=X(:,ind((j-1)*batch_size+1:j*batch_size));            % fetch the batch and the corresponding Laplacian sub-matrix
            s_G=SG(ind((j-1)*batch_size+1:j*batch_size),ind((j-1)*batch_size+1:j*batch_size));
            % Forward ----
            [x_e,z,z_d,x_hat] = forward_network(x,We1,be1,We2,be2,Wd1,bd1,Wd2,bd2,network_setup);
            
            res=sqrt(sum((x-x_hat).^2));
            
            % Backward ----
            
            t = t + 1;
            [We1,be1,We2,be2,Wd1,bd1,Wd2,bd2, ...
                m_e1, v_e1, m_e1b, v_e1b, m_e2, v_e2, m_e2b, v_e2b, ...
                m_d1, v_d1, m_d1b, v_d1b, m_d2, v_d2, m_d2b, v_d2b, grad] ...
                =trainAE(x,x_e,z,z_d,x_hat,res,s_G,lr,lamda, ...
                We1,be1,We2,be2,Wd1,bd1,Wd2,bd2,network_setup, ...
                m_e1, v_e1, m_e1b, v_e1b, m_e2, v_e2, m_e2b, v_e2b, ...
                m_d1, v_d1, m_d1b, v_d1b, m_d2, v_d2, m_d2b, v_d2b, t);
            
%             error_curve(end + 1) = sum(sum(grad))/(n*L);
            error_curve(end + 1) = sum(res)/(n);
%             epoch_loss = epoch_loss + sum(sum(grad))/(n*L);
            epoch_loss = epoch_loss + sum(res)/(n);
        end
        epoch_loss_curve(end + 1) = epoch_loss;
        % For every epoch:
        string = ['Epoch number: ', num2str(epoch), '.']; 
        disp(string);
        
        [X_e_test,Z_test,X_d_test,X_hat_test] ...
            = forward_network(X,We1,be1,We2,be2,Wd1,bd1,Wd2,bd2,network_setup);

        y_test=sum((X-X_hat_test).^2);
        
        zips=[idex,y_test'];            % Recover the original image with 'idex'
        zips_sort=sortrows(zips,1);
        y_test=zips_sort(:,2);
        
        y_test=reshape(y_test,size(map,1),size(map,2));
        
        AUC = ROC(y_test,map,0);
        All_AUC(end + 1) = AUC;
        prev_AUC = All_AUC(end);
        AUC_change = abs(AUC - prev_AUC);
        
        es = true;
        n_loss = 10;
        if length(epoch_loss_curve) > n_loss 
            last_loss = epoch_loss_curve(end-1);
            
            last_n_elements = epoch_loss_curve(end-n_loss+1:end);
            
            last_n_std = std(last_n_elements);
        else
            last_n_std = 1;
        end
        
        
        if es == true
%             if epoch_loss < 0.3 && last_n_std < 0.01
            if epoch_loss < 0.4 && last_n_std < 0.01
                
                disp(string);                
                break
            else
                AUC_values = [AUC_values(2:end)];
                AUC_values(end + 1) = AUC;
            end
        end
        
        
        
    end
    
    plot_AUC = false;
    if plot_AUC == true
        L = length(All_AUC);
        empty = 1:L;
        plot(empty, All_AUC);
    end
    
    L = length(All_AUC);
    empty = 1:L;
    subplot(3,1,1);
    plot(empty, All_AUC);
    title('AUC for each epoch');

    subplot(3,1,2);
    plot(empty, epoch_loss_curve);
    title('Loss for each epoch');

    L = length(error_curve);
    empty = 1:L;
    subplot(3,1,3);
    plot(empty, error_curve);
    title('Loss for each batch iteration');
    
    % Specify the folder path
    folder_path = join(['results_master/', dataset_name]);

    % Check if the folder exists
    if ~exist(folder_path, 'dir')
        % If the folder does not exist, create it using the mkdir function
        mkdir(folder_path);
    end
    
    saveas(gcf, join([folder_path, '/', network_setup, '_', int2str(hid), '.png']));
    close(gcf);

    % Output
    [X_e,Z,X_d,X_hat] ...
            = forward_network(X,We1,be1,We2,be2,Wd1,bd1,Wd2,bd2,network_setup);
    y=sum((X-X_hat).^2);

end

function y=sigmoid(W,x,b)
% Sigmoid function
    y=1./(1+exp(-(W*x+repmat(b,[1,size(x,2)]))));
end

function [x_e,z,z_d,x_hat]=forward_network(x,We1,be1,We2,be2,Wd1,bd1,Wd2,bd2,network_setup)
    if 'sigsig' == network_setup
            % Encoder:
            x_e=sigmoid(We1,x,be1);
            z=sigmoid(We2,x_e,be2);
            
            % Decoder:
            z_d=sigmoid(Wd1,z,bd1);
            x_hat=sigmoid(Wd2,z_d,bd2);
    elseif 'defsig' == network_setup
        % Encoder:
        x_e = We1*x+be1;
        z=sigmoid(We2,x_e,be2);

        % Decoder:
        z_d = Wd1 * z + bd1;
        x_hat=sigmoid(Wd2,z_d,bd2);
    elseif 'relsig' == network_setup
        % Encoder:
        x_e = max(We1*x+be1,0);
        z=sigmoid(We2,x_e,be2);

        % Decoder:
        z_d = max(Wd1 * z + bd1,0);
        x_hat=sigmoid(Wd2,z_d,bd2);
    end
end

function [We1_,be1_,We2_,be2_,Wd1_,bd1_,Wd2_,bd2_, ...
     m_e1_, v_e1_, m_e1b_, v_e1b_, m_e2_, v_e2_, m_e2b_, v_e2b_, ...
    m_d1_, v_d1_, m_d1b_, v_d1b_, m_d2_, v_d2_, m_d2b_, v_d2b_, grad] ...
    =trainAE(x,x_e,z,z_d,x_hat,res,L,lr,lamda, ...
    We1,be1,We2,be2,Wd1,bd1,Wd2,bd2,network_setup, ...
    m_e1, v_e1, m_e1b, v_e1b, m_e2, v_e2, m_e2b, v_e2b, ...
    m_d1, v_d1, m_d1b, v_d1b, m_d2, v_d2, m_d2b, v_d2b, t)
    % Training the network by BP
    [n,num]=size(x);
    D=repmat(1./(2*res),n,1);
    grad=(x_hat-x).*D; % Error 
    
    if 'sigsig' == network_setup
    
        deltad1=grad.*x_hat.*(1-x_hat); % Derivate of sigmoid for Wd2/bd2
        deltad2=Wd2'*deltad1.*(z_d.*(1-z_d)); % Derivate of sigmoid for Wd1/bd1

        delta_L=z*(L+L');

        deltae1 = (Wd1'*deltad2+lamda*delta_L).*(z.*(1-z)); % Derivate of sigmoid for We2/be2
        deltae2 = We2'*deltae1.*(x_e.*(1-x_e)); % Derivate of sigmoid for We1/be1
        
    elseif 'defsig' == network_setup
            
        deltad1=grad.*x_hat.*(1-x_hat); % Derivate of sigmoid and layer d2
        deltad2=Wd2'*deltad1; % Derivate of layer d1

        delta_L=z*(L+L');

        deltae1 = (Wd1'*deltad2+lamda*delta_L).*(z.*(1-z)); % Derivate of sigmoid and layer e2
        deltae2 = We2'*deltae1; % Derivate of layer e1
    
    elseif 'relsig' == network_setup
            
        deltad1=grad.*x_hat.*(1-x_hat); % Derivate of sigmoid and layer d2
        
        relu_der_z_d = z_d;
        relu_der_z_d(relu_der_z_d > 0) = 1;
        deltad2=Wd2'*deltad1.*relu_der_z_d; % Derivate of layer d1

        delta_L=z*(L+L');

        deltae1 = (Wd1'*deltad2+lamda*delta_L).*(z.*(1-z)); % Derivate of sigmoid and layer e2
        relu_der_x_e = x_e;
        relu_der_x_e(relu_der_x_e > 0) = 1;
        deltae2 = We2'*deltae1.*relu_der_x_e; % Derivate of layer e1
    end
    
    % Updating weights
    
    % ADAM parameters
    epsilon = 0.00001;
    beta1 = 0.9;
    beta2 = 0.99;
    
    optimizer = "adam";
    
    if optimizer == "adam"
        % Decoder        
        [Wd2_, m_d2_, v_d2_] = adamupdate(Wd2, deltad1*z_d'/num, m_d2, v_d2, t, lr, beta1, beta2, epsilon);
        [bd2_, m_d2b_, v_d2b_] = adamupdate(bd2, deltad1*ones(1,num)'/num, m_d2b, v_d2b, t, lr, beta1, beta2, epsilon);
        [Wd1_, m_d1_, v_d1_] = adamupdate(Wd1, deltad2*z'/num, m_d1, v_d1, t, lr, beta1, beta2, epsilon);
        [bd1_, m_d1b_, v_d1b_] = adamupdate(bd1, deltad2*ones(1,num)'/num, m_d1b, v_d1b, t, lr, beta1, beta2, epsilon);

        % Encoder
        [We2_, m_e2_, v_e2_] = adamupdate(We2, deltae1*x_e'/num, m_e2, v_e2, t, lr, beta1, beta2, epsilon);
        [be2_, m_e2b_, v_e2b_] = adamupdate(be2, deltae1*ones(1,num)'/num, m_e2b, v_e2b, t, lr, beta1, beta2, epsilon);
        [We1_, m_e1_, v_e1_] = adamupdate(We1, deltae2*x'/num, m_e1, v_e1, t, lr, beta1, beta2, epsilon);
        [be1_, m_e1b_, v_e1b_] = adamupdate(be1, deltae2*ones(1,num)'/num, m_e1b, v_e1b, t, lr, beta1, beta2, epsilon);
        
    elseif optimizer == "none"
        % Decoder
        Wd2_ = Wd2 - lr*deltad1*z_d'/num;
        bd2_ = bd2 - lr*deltad1*ones(1,num)'/num;
        Wd1_ = Wd1 - lr*deltad2*z'/num;
        bd1_ = bd1 - lr*deltad2*ones(1,num)'/num;

        % Encoder
        We2_ = We2 - lr*deltae1*x_e'/num;
        be2_ = be2 - lr*deltae1*ones(1,num)'/num;
        We1_ = We1 - lr*deltae2*x'/num;
        be1_ = be1 - lr*deltae2*ones(1,num)'/num;
    end
end