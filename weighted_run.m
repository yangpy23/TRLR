function [total_ret, day_ret] = weighted_run(data,tc)
%{
This function is the main function of Trend Representation Based 
Log-density Regularization(TRLR).
Please note that TRLR is protected by patent. Any commercial or industrial
purposes of using TRLR are limited. But it is encouraged for usage of
study and research.

For any usage of this function, the following papers should be cited as reference, 
where parameter settings and details are given in the first paper:

[1] Pei-Yi Yang, Zhao-Rong Lai, Xiaotian Wu, Liangda Fang. ¡°Trend Representation 
Based Log-density Regularization System for Portfolio Optimization¡±,  
Pattern Recognition, vol. 76, pp. 14-24, Apr. 2018.
[2] Zhao-Rong Lai, Dao-Qing Dai, Chuan-Xian Ren, and Ke-Kun Huang. "A Peak Price 
Tracking-Based Learning System for Portfolio Selection", 
IEEE Transactions on Neural Networks and Learning Systems, PP(99):1-10, Jun, 2017.

Inputs:
     data            - market sequence vectors, TxN
     tc              - transaction cost rate

Ouputs:
     total_ret       - total return, Tx1
     day_ret         - daily return, Tx1

%}

W=5;
[T N]=size(data);

% Return Variables
run_ret =1;
total_ret = ones(T, 1);
day_ret = ones(T, 1);

% Portfolio variables, starting with uniform portfolio
day_weight = ones(N, 1)/N;  
day_weight_total=ones(N, T)/N;
day_weight_o = zeros(N, 1);
day_weight_n = zeros(N, 1);
turno = 0;

%to get the close price according to relative price
data_close = ones(T,N);
for i=2:T
    data_close(i,:)= data_close(i-1,:).*data(i,:);
end

for t = 5:1:T
    
    %Calculate return and total return at the end of the t-th day.
    day_weight_total(:,t)=day_weight;
    day_ret(t, 1) = (data(t, :)*day_weight)*(1-tc/2*sum(abs(day_weight-day_weight_o)));

    run_ret = run_ret * day_ret(t, 1);
    total_ret(t, 1) = run_ret;
    
    % Adjust weight(t, :) for the transaction cost issue
    day_weight_o = day_weight.*data(t, :)'/(data(t, :)*day_weight);

    %Update portfolio
     if(t<T)
        [day_weight_n]=weighted_dayweight_ridge(data_close,data,t+1,day_weight, W);
        
       turno = turno + 0.5*sum(abs(day_weight_n-day_weight));
       day_weight = day_weight_n;
    
     end
end
if total_ret(end)<10000
    fprintf('\t %f \n',total_ret(end));
else
    fprintf('\t %e \n',total_ret(end));
end
end

