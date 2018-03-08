function w = simplex_projection_selfnorm2(v, b)
%{
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
%}

%v = (v > 0) .* v;
while(max(abs(v))>1e6)
v=v/10;
end

u = sort(v,'descend');



sv = cumsum(u);
rho = find(u > (sv - b) ./ (1:length(u))', 1, 'last');
theta = (sv(rho) - b) / rho;
w = max(v - theta, 0);
end
