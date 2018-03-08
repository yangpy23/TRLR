function [day_weight]=weighted_dayweight_ridge(data_close,data,t1,day_weight, w)
%{
This function is the sub function to calculate the day_weight at the t+1 day with TRLR
Please note that TRLR is protected by patent. Any commercial or industrial
purposes of using TRLR are limited. But it is encouraged for usage of
study and research.

For any usage of this function, the following papers should be cited as reference, 
where parameter settings and details are given in the first paper:

[1] Pei-Yi Yang, Zhao-Rong Lai, Xiaotian Wu, Liangda Fang. “Trend Representation 
Based Log-density Regularization System for Portfolio Optimization”,  
Pattern Recognition, vol. 76, pp. 14-24, Apr. 2018.
[2] Zhao-Rong Lai, Dao-Qing Dai, Chuan-Xian Ren, and Ke-Kun Huang. "A Peak Price 
Tracking-Based Learning System for Portfolio Selection", 
IEEE Transactions on Neural Networks and Learning Systems, PP(99):1-10, Jun, 2017.

Inputs:
   data_close             -the close price data
   data                   -the price relative data
   t1                     -the new day 
   day_weight             -the weight before the new day
   w                      -the window size
Output:
   day_weight             -the weight at the new day
%}

% parameter setting
q1 = 0.1;
lambda = 0.7;
eta=40000;
sigma=0.3;

[~,N]=size(data);
Kernel=zeros(N,N);
Rpredict=zeros(1,N);

if t1<w+2
    x_t1=data(t1-1,:);
else
    Rbefore=data_close((t1-w):(t1-1),:);  
    Rmax=max(Rbefore);

    %weighted ridge regression
    
    d = 2*(1-q1)/(w-1); % common difference
    W = diag( q1:d:q1+(w-1)*d );
    X = [ones(w,1)  (1 :w)'];
    Y=Rbefore;
    wt = (X'*W*X+lambda*eye(2))\(X'*W*Y);
    Rpredict= [1 w+1]*wt;
    Rpredict = (Rpredict+Rmax)/2;

    x_t1 = Rpredict./data_close(t1-1,:); % future price relative prediction
end 

%%--------------optimization-----------

    x_t1 = x_t1';
    x_t1_norm = x_t1 - mean(x_t1);

    for i = 1:N
        for j = 1:N
            Kernel(i,j) = exp(-(x_t1(i)-x_t1(j))^2/(2*sigma^2));
        end
    end
    
    day_weight = day_weight + eta*Kernel*x_t1_norm;

day_weight = simplex_projection_selfnorm2(day_weight,1); % projection
end

