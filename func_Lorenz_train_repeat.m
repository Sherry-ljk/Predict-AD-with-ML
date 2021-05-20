function mean_rmse = func_Lorenz_train_repeat(hyperpara_set,n,repeat_num)
tic
beta = hyperpara_set(1);
eig_rho = hyperpara_set(2);
k = hyperpara_set(3);
W_in_a = hyperpara_set(4);
alpha = hyperpara_set(5);
p_0 = hyperpara_set(6);
k_p = hyperpara_set(7);
Nt = 60000;%training length
Np = 400;%predicting length
wa = 100;%warmup length
transit = 50;%abondon reservoir length 
n = 1000;%reservoir size
dim = 6; %system dimension
tp_dim=3;%training parameter dimension
k = 20;%mean degree of W_res
p=[0.21,0.22,0.23];
x= importdata('x.mat');
rmse_set = zeros(repeat_num,1);
parfor repeat_i = 1:repeat_num
    rmse_total = zeros(dim,Np);
    rmse_mean = zeros(tp_dim,1);     
    rng(repeat_i*20000 + (now*1000-floor(now*1000))*100000)
% W_in n*dim
W_in=zeros(n,dim+1);
for i=1:n
     W_in(i,ceil(i*dim/n))=(2*rand()-1)*W_in_a;  
end
W_in(:,dim+1)=(2*rand(n,1)-1)*W_in_a;
% ER network n*n and its radius is eig_rho 
p1 = k/(n-1);
W = zeros(n,n);
for i=1:n
    for j=1:n
        b=rand();
       if (i~=j)&&(b<p1)          
            W(i,j)=rand(); 
       end  
    end
end
rad = max(abs(eig(W)));
W_res = W*(eig_rho/rad);
W_res = sparse(W_res);
%%%%%training phase
U = zeros(dim,tp_dim*(Nt-transit+1));
R = zeros(n,tp_dim*(Nt-transit+1));
for ii=1:tp_dim
    m = randperm(2000,1);
    u_train = x(6*ii-5:6*ii,m:m+Nt);
    pp = (p(ii)-p_0)*k_p*ones(1,Nt+1);
    u1_train = [u_train;pp];
    r1 = zeros(n,Nt+1);
    r2 = zeros(n,Nt+1);
    for i=1:Nt
        r1(:,i+1)=(1-alpha)*r1(:,i)+alpha*tanh(W_res*r1(:,i)+W_in*u1_train(:,i));
        r2(:,i+1)=r1(:,i+1);
        r2(2:2:n,i+1)=r1(2:2:n,i+1).^2;    
    end
 U(:,(Nt-transit+1)*(ii-1)+1:(Nt-transit+1)*ii)=u1_train(1:dim,transit+1:Nt+1);
 R(:,(Nt-transit+1)*(ii-1)+1:(Nt-transit+1)*ii)=r2(:,transit+1:Nt+1); 
end
I = eye(n);
W_out=[(U(:,:))]*(R(:,:)')*inv((R(:,:)*(R(:,:)')+beta*I));
%%%%% predicting phase
 mm=randperm(3000,1)+15000;
for jj=1:tp_dim
    u_train = x(6*jj-5:6*jj,mm+1:mm+Np);
    pp = (p(jj)-p_0)*k_p*ones(1,Np);
    u_predict = [u_train;pp];
    r3 = zeros(n,Np);
    r4 = zeros(n,1);
    for i=1:Np-1
        r3(:,i+1)=(1-alpha)*r3(:,i)+alpha*tanh(W_res*r3(:,i)+W_in*u_predict(:,i));
        r4(:)=r3(:,i+1);
        r4(2:2:n)=r3(2:2:n,i+1).^2; 
        if (i>=wa) 
           u_predict(1:dim,i+1)=W_out*r4(:);
        end
    end
rmse_total(:,:)=u_train(1:dim,1:Np)-u_predict(1:dim,1:Np);
rmse_mean(jj)=sqrt(mean(abs(rmse_total(:)).^2));
end 
rmse_set(repeat_i)=max(rmse_mean);
end
mean_rmse = mean(rmse_set);
fprintf('\nmean rmse is %f\n',mean_rmse)
toc
end
