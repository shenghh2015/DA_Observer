DA_folder = 'mmd-0.8-lr-0.0001-bz-400-iter-50000-scr-True-shar-True-fc-128-bn-False-tclf-0.0-sclf-1.0-tlabels-0-vclf-1-total-val-100';
TF_folder = 'TF-lr-1e-06-bz-50-iter-50000-scr-False-fc-128-bn-False-trg_labels-70-clf_v1-total';
source_folder = 'cnn-4-bn-False-noise-2.0-trn-100000-sig-0.035-bz-400-lr-5e-05-Adam-100.0k';

DA_stats = load(fullfile(DA_folder, 'roc_fit_output.txt'));
TF_stats = load(fullfile(TF_folder, 'roc_fit_output.txt'));
source_stats = load(fullfile(source_folder, 'roc_fit_output.txt'));

DA_AUC = load(fullfile(DA_folder, 'fit_auc.txt'));
TF_AUC = load(fullfile(TF_folder, 'fit_auc.txt'));
soruce_AUC = load(fullfile(source_folder, 'fit_auc.txt'));

figure(1)
linewidth = 2.5;
FontSize = 16;
plot(DA_stats(:,1), DA_stats(:,2), 'b-','linewidth', linewidth);
hold on;
plot(TF_stats(:,1), TF_stats(:,2), 'r-.', 'linewidth', linewidth);
hold on;
plot(source_stats(:,1), source_stats(:,2), 'r--', 'linewidth', linewidth);
hold on;

xlabel('FPF');
ylabel('TPF')
legend(['AUC_{DA}:',num2str(DA_AUC(1)),'\pm',num2str(DA_AUC(2))],...
     ['AUC_{TL-140}:',num2str(TF_AUC(1)),'\pm',num2str(TF_AUC(2))],...
     ['AUC_{SO}:',num2str(soruce_AUC(1)),'\pm',num2str(soruce_AUC(2))])   
set(findall(gcf,'-property','FontSize'),'FontSize',FontSize)
