
clear
close all



load epsilon01_Housing_MinMax_GaussianDP_delta01_v2.mat loss_GD loss_DP_GD_1 loss_DP_GD_10 loss_fixedSeed_DP_GD...
    loss_fixedSeed_DP_GD_1 loss_fixedSeed_DP_GD_10

markerStep = 100000;


figure(1);
subplot(1,3,1)
semilogy(loss_GD,'MarkerIndices',1:markerStep:length(loss_GD),'LineWidth',2);
hold on
semilogy(loss_DP_GD_1,'--*','MarkerIndices',1:markerStep:length(loss_DP_GD_1),'LineWidth',2);
semilogy(loss_DP_GD_10,'--^','MarkerIndices',1:markerStep:length(loss_DP_GD_10),'LineWidth',2);
semilogy(loss_fixedSeed_DP_GD,':*r','MarkerIndices',1:markerStep:length(loss_fixedSeed_DP_GD),'LineWidth',2);
semilogy(loss_fixedSeed_DP_GD_1,'-.s','MarkerIndices',1:markerStep:length(loss_fixedSeed_DP_GD_1),'LineWidth',2);
semilogy(loss_fixedSeed_DP_GD_10,'-.x','MarkerIndices',1:markerStep:length(loss_fixedSeed_DP_GD_10),'LineWidth',2);
xlabel({'Number of Iterations';'(a)'},'fontsize',16,'fontname','Times New Roman')
ylabel('Loss','fontsize',16,'fontname','Times New Roman')
legend('FedGD','DP-FedGD, c=1','DP-FedGD, c=10','Proposed-FedGD','Proposed-FedGD, c=1','Proposed-FedGD, c=10');
title('\epsilon=0.1, \delta=0.1')
set(gca,'fontsize',14,'fontweight','bold');



load epsilon05_Housing_MinMax_GaussianDP_delta01_v2.mat loss_GD loss_DP_GD_1 loss_DP_GD_10 loss_fixedSeed_DP_GD...
    loss_fixedSeed_DP_GD_1 loss_fixedSeed_DP_GD_10

subplot(1,3,2)
semilogy(loss_GD,'MarkerIndices',1:markerStep:length(loss_GD),'LineWidth',2);
hold on
semilogy(loss_DP_GD_1,'--*','MarkerIndices',1:markerStep:length(loss_DP_GD_1),'LineWidth',2);
semilogy(loss_DP_GD_10,'--^','MarkerIndices',1:markerStep:length(loss_DP_GD_10),'LineWidth',2);
semilogy(loss_fixedSeed_DP_GD,':*r','MarkerIndices',1:markerStep:length(loss_fixedSeed_DP_GD),'LineWidth',2);
semilogy(loss_fixedSeed_DP_GD_1,'-.s','MarkerIndices',1:markerStep:length(loss_fixedSeed_DP_GD_1),'LineWidth',2);
semilogy(loss_fixedSeed_DP_GD_10,'-.x','MarkerIndices',1:markerStep:length(loss_fixedSeed_DP_GD_10),'LineWidth',2);
xlabel({'Number of Iterations';'(b)'},'fontsize',16,'fontname','Times New Roman')
ylabel('Loss','fontsize',16,'fontname','Times New Roman')
legend('FedGD','DP-FedGD, c=1','DP-FedGD, c=10','Proposed-FedGD','Proposed-FedGD, c=1','Proposed-FedGD, c=10');
title('\epsilon=0.5, \delta=0.1')
set(gca,'fontsize',14,'fontweight','bold');



load epsilon1_Housing_MinMax_GaussianDP_delta01_v2.mat loss_GD loss_DP_GD_1 loss_DP_GD_10 loss_fixedSeed_DP_GD...
    loss_fixedSeed_DP_GD_1 loss_fixedSeed_DP_GD_10


subplot(1,3,3)
semilogy(loss_GD,'MarkerIndices',1:markerStep:length(loss_GD),'LineWidth',2);
hold on
semilogy(loss_DP_GD_1,'--*','MarkerIndices',1:markerStep:length(loss_DP_GD_1),'LineWidth',2);
semilogy(loss_DP_GD_10,'--^','MarkerIndices',1:markerStep:length(loss_DP_GD_10),'LineWidth',2);
semilogy(loss_fixedSeed_DP_GD,':*r','MarkerIndices',1:markerStep:length(loss_fixedSeed_DP_GD),'LineWidth',2);
semilogy(loss_fixedSeed_DP_GD_1,'-.s','MarkerIndices',1:markerStep:length(loss_fixedSeed_DP_GD_1),'LineWidth',2);
semilogy(loss_fixedSeed_DP_GD_10,'-.x','MarkerIndices',1:markerStep:length(loss_fixedSeed_DP_GD_10),'LineWidth',2);
xlabel({'Number of Iterations';'(c)'},'fontsize',16,'fontname','Times New Roman')
ylabel('Loss','fontsize',16,'fontname','Times New Roman')
legend('FedGD','DP-FedGD, c=1','DP-FedGD, c=10','Proposed-FedGD','Proposed-FedGD, c=1','Proposed-FedGD, c=10');
title('\epsilon=1, \delta=0.1')
set(gca,'fontsize',14,'fontweight','bold');

