
clear
close all


%% data allocation for linear regression
[Xdata] = csvread('data_framingham_clean/x_train_MinMax_Normalized.csv',1,0); % California housing dataset
[ydata] = csvread('data_framingham_clean/y_train_MinMax_Normalized.csv',1,0); 

% [Xdata] = csvread('data_framingham_clean/x_train_zscore_Normalized.csv',1,0); % California housing dataset
% [ydata] = csvread('data_framingham_clean/y_train_zscore_Normalized.csv',1,0); 
 % 
 

num_feature=size(Xdata,2);
total_sample=size(Xdata,1);
num_workers=10;
per_split=floor(total_sample/num_workers);
total_sample =num_workers*per_split;
num_sample=per_split;


%num_iter=2000;
X=cell(num_workers);
y=cell(num_workers);



for n=1:num_workers
        first = (n-1)*per_split+1;
        last = first+per_split-1;
        X{n}=Xdata(first:last,1:num_feature);
        y{n}=ydata(first:last);
end


num_feature=size(X{1},2);

X_fede=[];
y_fede=[];
for i=1:num_workers
  X_fede=[X_fede;X{i}];
  y_fede=[y_fede;y{i}];
end

lambda=1E-4;

%lambda=1E-4;

for i=1:length(y_fede)
    if(y_fede(i)==0)
        y_fede(i)=-1;
    end
end

%% Optimal solution

XX=X_fede;
YY=y_fede;

obj0=opt_sol_logistic(XX,YY, num_feature, lambda, num_workers, num_sample)

%obj0=0.5864;
%obj0=0.3823;

num_iter=1000000;
acc = 1E-10;

%rho = 1;
%lr=1E-5;
lr=1E-1;
epsilon=0.1;
delta=0.1;


[obj_GD, loss_GD, Iter_GD] = distributed_GD_logReg(X_fede,y_fede, num_workers, num_feature, num_sample, num_iter, obj0...
    , acc, lr, lambda);

c=1;



[obj_DP_GD_1, loss_DP_GD_1, Iter_DP_GD_1] = DP_GD_logReg(X_fede,y_fede, num_workers, num_feature, num_sample, num_iter, obj0...
    , acc, lr,epsilon, delta, c, lambda);



c=10;

[obj_DP_GD_10, loss_DP_GD_10, Iter_DP_GD_10] = DP_GD_logReg(X_fede,y_fede, num_workers, num_feature, num_sample, num_iter, obj0...
    , acc, lr,epsilon, delta, c, lambda);



c=10000;

[obj_fixedSeed_DP_GD, loss_fixedSeed_DP_GD, Iter_fixedSeed_DP_GD] = fixedSeed_DP_GD_withClipping_logReg(X_fede,y_fede, num_workers, num_feature, num_sample, num_iter, obj0...
    , acc, lr, epsilon, delta, c, lambda);


c=1;

[obj_fixedSeed_DP_GD_1, loss_fixedSeed_DP_GD_1, Iter_fixedSeed_DP_GD_1] = fixedSeed_DP_GD_withClipping_logReg(X_fede,y_fede, num_workers, num_feature, num_sample, num_iter, obj0...
    , acc, lr,epsilon, delta, c, lambda);



c=10;

[obj_fixedSeed_DP_GD_10, loss_fixedSeed_DP_GD_10, Iter_fixedSeed_DP_GD_10] = fixedSeed_DP_GD_withClipping_logReg(X_fede,y_fede, num_workers, num_feature, num_sample, num_iter, obj0...
    , acc, lr,epsilon, delta, c, lambda);




markerStep=100000;
figure(1);
semilogy(loss_GD,'MarkerIndices',1:markerStep:length(loss_GD),'LineWidth',2);
hold on
semilogy(loss_DP_GD_1,'--*','MarkerIndices',1:markerStep:length(loss_DP_GD_1),'LineWidth',2);
semilogy(loss_DP_GD_10,'--^','MarkerIndices',1:markerStep:length(loss_DP_GD_10),'LineWidth',2);

semilogy(loss_fixedSeed_DP_GD,':*r','MarkerIndices',1:markerStep:length(loss_fixedSeed_DP_GD),'LineWidth',2);
semilogy(loss_fixedSeed_DP_GD_1,'-.s','MarkerIndices',1:markerStep:length(loss_fixedSeed_DP_GD_1),'LineWidth',2);
semilogy(loss_fixedSeed_DP_GD_10,'-.x','MarkerIndices',1:markerStep:length(loss_fixedSeed_DP_GD_10),'LineWidth',2);

xlabel({'Number of Iterations';'(a)'},'fontsize',16,'fontname','Times New Roman')
ylabel('Loss','fontsize',16,'fontname','Times New Roman')
%legend('FedGD','DP-FedGD-c=1','DP-FedGD-c=10','Proposed-FedGD','Proposed-FedGD-c=1','Proposed-FedGD-c=10');
legend('FedGD','DP-FedGD-c=1','DP-FedGD-c=10','Proposed-FedGD','Proposed-FedGD-c=1','Proposed-FedGD-c=10');

title('\epsilon=0.1')
set(gca,'fontsize',14,'fontweight','bold');


save epsilon01_logReg_MinMax_GaussianDP_v2.mat loss_GD loss_DP_GD_1 loss_DP_GD_10 loss_fixedSeed_DP_GD...
    loss_fixedSeed_DP_GD_1 loss_fixedSeed_DP_GD_10
