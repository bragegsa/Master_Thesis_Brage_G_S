function y=myRGAE_ADAM_multi(X,SG,lamda,hid, map, epochs, lr, idex)%
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
    epoch_loss_curve = [];
    
    for epoch=1:epochs
        ind=randperm(n);
        epoch_loss = 0;
        for j=1:batch_num
            x=X(:,ind((j-1)*batch_size+1:j*batch_size));            % fetch the batch and the corresponding Laplacian sub-matrix
            s_G=SG(ind((j-1)*batch_size+1:j*batch_size),ind((j-1)*batch_size+1:j*batch_size));
            % Forward ----
            
            % Encoder:
            x_e=sigmoid(We1,x,be1);
            z=sigmoid(We2,x_e,be2);
%             z = We2 * x_e + be2;
%             z = relu(We2,x_e,be2);
            
            % Decoder:
            z_d=sigmoid(Wd1,z,bd1);
%             z_d = Wd1 * z + bd1;
%             z_d = relu(Wd1,z,bd1);
            x_hat=sigmoid(Wd2,z_d,bd2);
            
            res=sqrt(sum((x-x_hat).^2));
            
            % Backward ----
            t = t + 1;
            
            [We1,be1,We2,be2,Wd1,bd1,Wd2,bd2,...
                m_e1, v_e1, m_e1b, v_e1b, m_e2, v_e2, m_e2b, v_e2b, ...
                m_d1, v_d1, m_d1b, v_d1b, m_d2, v_d2, m_d2b, v_d2b] ....
                =trainAE(x,x_e,z,z_d,x_hat,res,s_G,lr,lamda, ...
                We1,be1,We2,be2,Wd1,bd1,Wd2,bd2, ...
                m_e1, v_e1, m_e1b, v_e1b, m_e2, v_e2, m_e2b, v_e2b, ...
                m_d1, v_d1, m_d1b, v_d1b, m_d2, v_d2, m_d2b, v_d2b, t);
            
            epoch_loss = epoch_loss + sum(res)/(n);
        end

        epoch_loss_curve(end + 1) = epoch_loss;
        % For every epoch:
        string = ['Epoch number: ', num2str(epoch), '.']; 
        disp(string);
        
        % Droput:
        X_e_test=sigmoid(We1,X,be1);
        Z_test=sigmoid(We2,X_e_test,be2);
%         Z_test=We2*X_e_test + be2;
%         Z_test = relu(We2,X_e_test,be2);
        
        X_d_test=sigmoid(Wd1,Z_test,bd1);
%         X_d_test=Wd1*Z_test + bd1;
%         X_d_test = relu(Wd1,Z_test,bd1);
        X_hat_test=sigmoid(Wd2,X_d_test,bd2);

        y_test=sum((X-X_hat_test).^2);
        
        zips=[idex,y_test'];            % Recover the original image with 'idex'
        zips_sort=sortrows(zips,1);
        y_test=zips_sort(:,2);
        
        y_test=reshape(y_test,size(map,1),size(map,2));
        
        AUC = ROC(y_test,map,0);
        disp(AUC);
        
        All_AUC(end + 1) = AUC;
        prev_AUC = All_AUC(end);
        AUC_change = abs(AUC - prev_AUC);
        
        n_loss = 100;
        if length(epoch_loss_curve) > n_loss 
%             last_loss = epoch_loss_curve(end-1);
%             
            last_n_elements = epoch_loss_curve(end-n_loss+1:end);
            
            last_n_std = std(last_n_elements);
        else
            last_n_std = 1;
        end
        
        dropout = true;
        if dropout == true
%             if (AUC < min(AUC_values) && epoch > 30) | (AUC_change < 0.0001 && epoch > 100 )
            if epoch_loss < 0.2 && last_n_std < 0.006
                string = ['Last AUC value was ', num2str(AUC), ' and the minimal of the last AUCs was ', num2str(min(AUC_values)), '.'];
                disp(string);
                break
            else
                AUC_values = [AUC_values(2:end)];
                AUC_values(end + 1) = AUC;
            end
        end
        
        
        
    end
    
    plot_AUC = true;
    if plot_AUC == true
        L = length(All_AUC);
        empty = 1:L;
        plot(empty, All_AUC);
    end

    % Output
    X_e=sigmoid(We1,X,be1);
    Z=sigmoid(We2,X_e,be2);
    X_d=sigmoid(Wd1,Z,bd1);
    X_hat=sigmoid(Wd2,X_d,bd2);
    y=sum((X-X_hat).^2);

end

function y=sigmoid(W,x,b)
% Sigmoid function
    y=1./(1+exp(-(W*x+repmat(b,[1,size(x,2)]))));
end

function y = relu(W, x, b)
    % Rectified Linear Unit (ReLU) function
    z = W * x + repmat(b, [1, size(x, 2)]);
    y = max(0, z);
end

function [We1_,be1_,We2_,be2_,Wd1_,bd1_,Wd2_,bd2_, ...
    m_e1_, v_e1_, m_e1b_, v_e1b_, m_e2_, v_e2_, m_e2b_, v_e2b_, ...
    m_d1_, v_d1_, m_d1b_, v_d1b_, m_d2_, v_d2_, m_d2b_, v_d2b_] ...
    =trainAE(x,x_e,z,z_d,x_hat,res,L,lr,lamda, ...
    We1,be1,We2,be2,Wd1,bd1,Wd2,bd2, ...
    m_e1, v_e1, m_e1b, v_e1b, m_e2, v_e2, m_e2b, v_e2b, ...
    m_d1, v_d1, m_d1b, v_d1b, m_d2, v_d2, m_d2b, v_d2b, t)
% Training the network by BP
    [n,num]=size(x);
    D=repmat(1./(2*res),n,1);
    grad=(x_hat-x).*D;
    delta1=grad.*x_hat.*(1-x_hat);
    delta2=z*(L+L');
    tmp=(Wd1'*Wd2'*delta1+lamda*delta2).*(z.*(1-z));
    
    % ADAM parameters
    epsilon = 0.00001;
    beta1 = 0.9;
    beta2 = 0.99;
    
    % Decoder
    [Wd2_, m_d2_, v_d2_] = adamupdate(Wd2, delta1*z_d'/num, m_d2, v_d2, t, lr, beta1, beta2, epsilon);
    [bd2_, m_d2b_, v_d2b_] = adamupdate(bd2, delta1*ones(1,num)'/num, m_d2b, v_d2b, t, lr, beta1, beta2, epsilon);
    [Wd1_, m_d1_, v_d1_] = adamupdate(Wd1, Wd2.'*delta1*z'/num, m_d1, v_d1, t, lr, beta1, beta2, epsilon);
    [bd1_, m_d1b_, v_d1b_] = adamupdate(bd1, Wd2.'*delta1*ones(1,num)'/num, m_d1b, v_d1b, t, lr, beta1, beta2, epsilon);

    % Encoder
    [We2_, m_e2_, v_e2_] = adamupdate(We2, tmp*x_e'/num, m_e2, v_e2, t, lr, beta1, beta2, epsilon);
    [be2_, m_e2b_, v_e2b_] = adamupdate(be2, tmp*ones(1,num)'/num, m_e2b, v_e2b, t, lr, beta1, beta2, epsilon);
    [We1_, m_e1_, v_e1_] = adamupdate(We1, We2.'*tmp*x'/num, m_e1, v_e1, t, lr, beta1, beta2, epsilon);
    [be1_, m_e1b_, v_e1b_] = adamupdate(be1, We2.'*tmp*ones(1,num)'/num, m_e1b, v_e1b, t, lr, beta1, beta2, epsilon);
end