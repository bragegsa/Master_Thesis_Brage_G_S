function y=myRGAE_multi(X,SG,lamda,hid, map, epochs, lr, idex)%
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
    
    x_size = We1*X(:,batch_size+1:2*batch_size); % Size of output after first encoding layer
    x_r = size(x_size, 1);
    hid_2 = ceil(hid/2);
%     hid_2 = hid;
    
    We2=0.01*rand(hid_2,x_r);be2=rand(hid_2,1);     % weights and biases for encoder
    
    Wd1=0.01*rand(x_r,hid_2);bd1=rand(x_r,1);       % weights and biases for decoder
    Wd2=0.01*rand(L,hid);bd2=rand(L,1);       % weights and biases for decoder    
    
    n_AUC = 10;
    AUC_values = zeros(1,n_AUC); % Holds the previous n_AUC AUC values for dropout
    All_AUC = [];
    
    for epoch=1:epochs
        ind=randperm(n);
        for j=1:batch_num
            x=X(:,ind((j-1)*batch_size+1:j*batch_size));            % fetch the batch and the corresponding Laplacian sub-matrix
            s_G=SG(ind((j-1)*batch_size+1:j*batch_size),ind((j-1)*batch_size+1:j*batch_size));
            % Forward ----
            
            % Encoder:
            x_e=sigmoid(We1,x,be1);
            z=sigmoid(We2,x_e,be2);
%             z = We2 * x_e + be2;
            
            % Decoder:
            z_d=sigmoid(Wd1,z,bd1);
%             z_d = Wd1 * z + bd1;
            x_hat=sigmoid(Wd2,z_d,bd2);
            
            res=sqrt(sum((x-x_hat).^2));
            
            % Backward ----
            [We1,be1,We2,be2,Wd1,bd1,Wd2,bd2]=trainAE(x,x_e,z,z_d,x_hat,res,s_G,lr,lamda, ...
                We1,be1,We2,be2,Wd1,bd1,Wd2,bd2);
        end

        % For every epoch:
        string = ['Epoch number: ', num2str(epoch), '.']; 
        disp(string);
        
        % Droput:
        X_e_test=sigmoid(We1,X,be1);
        Z_test=sigmoid(We2,X_e_test,be2);
%         Z_test=We2*X_e_test + be2;
        X_d_test=sigmoid(Wd1,Z_test,bd1);
%         X_d_test=Wd1*Z_test + bd1;
        X_hat_test=sigmoid(Wd2,X_d_test,bd2);

        y_test=sum((X-X_hat_test).^2);
        
        zips=[idex,y_test'];            % Recover the original image with 'idex'
        zips_sort=sortrows(zips,1);
        y_test=zips_sort(:,2);
        
        y_test=reshape(y_test,size(map,1),size(map,2));
        
        AUC = ROC(y_test,map,0);
        All_AUC(end + 1) = AUC;
        prev_AUC = All_AUC(end);
        AUC_change = abs(AUC - prev_AUC);
        
        dropout = false;
        if dropout == true
            if (AUC < min(AUC_values) && epoch > 30) | (AUC_change < 0.00001 && epoch > 100 )
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

function [We1_,be1_,We2_,be2_,Wd1_,bd1_,Wd2_,bd2_] ...
    =trainAE(x,x_e,z,z_d,x_hat,res,L,lr,lamda, ...
    We1,be1,We2,be2,Wd1,bd1,Wd2,bd2)
% Training the network by BP
    [n,num]=size(x);
    D=repmat(1./(2*res),n,1);
    grad=(x_hat-x).*D;
    delta1=grad.*x_hat.*(1-x_hat);
    delta2=z*(L+L');
    tmp=(Wd1'*Wd2'*delta1+lamda*delta2).*(z.*(1-z));
    
    % Decoder
    Wd2_ = Wd2 - lr*delta1*z_d'/num;
    bd2_ = bd2 - lr*delta1*ones(1,num)'/num;
    Wd1_ = Wd1 - lr*Wd2.'*delta1*z'/num;
    bd1_ = bd1 - lr*Wd2.'*delta1*ones(1,num)'/num;

    % Encoder
    We2_ = We2 - lr*tmp*x_e'/num;
    be2_ = be2 - lr*tmp*ones(1,num)'/num;
    We1_ = We1 - lr*We2.'*tmp*x'/num;
    be1_ = be1 - lr*We2.'*tmp*ones(1,num)'/num;
end

function y = sigmoid_derivative(x)
    y = sigmoid_train(x) .* (1 - sigmoid_train(x));
end

function y = sigmoid_train(x)
    y = 1 ./ (1 + exp(-x));
end
