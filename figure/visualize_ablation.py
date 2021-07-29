# !/usr/bin/env python
# created by Yun Hao @MooreLab 2019
# This script contains codes that visualize ROC curves comparison across models ablation analysis.


## Module
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

## This function computes confidence interval of metric by bootstrapping.
def compute_metric_ci_by_bootsrap(metric_function, label_vec, pred_vec, confidence_interval = 0.95, bootstrap_times = 1000):
	## 0. Input arguments: 
		# metric_function:
		# label_vec: input true label array
		# pred_vec: input prediction probability array
		# confidence_interval: confidence interval to be computed (number between 0 and 1)
		# bootstrap_times: repeated sampling times for bootstrap

	## 1. Compute confidence interval of mean by bootstrapping
	vec_len = len(pred_vec)
	id_vec = np.arange(0, vec_len)
	# Repeat boostrap process
	sample_metrics = []
	np.random.seed(0)
	for sample in range(0, bootstrap_times):
		# Sampling with replacement from the input array
		sample_ids = np.random.choice(id_vec, size = vec_len, replace = True)
		sample_ids = np.unique(sample_ids)
		# compute sample metric
		sample_metric = metric_function(label_vec[sample_ids], pred_vec[sample_ids])
		sample_metrics.append(sample_metric)
	# sort means of bootstrap samples 
	sample_metrics = np.sort(sample_metrics)
	# obtain upper and lower index of confidence interval
	lower_id = int((0.5 - confidence_interval/2) * bootstrap_times) - 1
	upper_id = int((0.5 + confidence_interval/2) * bootstrap_times) - 1
	ci = (sample_metrics[upper_id] - sample_metrics[lower_id])/2

	return ci

## 0. Input and output files
plot_folder		= 'plot/ablation/performance_comparison_'
# 
full_pred_file		= 'prediction/29.tsv' 
no_struture_pred_file	= 'prediction/ablation-analysis/29_no_maccs.tsv'
no_gene_pred_file	= 'prediction/ablation-analysis/29_no_genes.tsv'
no_assay_pred_file	= 'prediction/ablation-analysis/29_no_assays.tsv'
plot_title		= 'tox21-pxr-p1'
# 
full_pred_file		= 'prediction/45.tsv'
no_struture_pred_file	= 'prediction/ablation-analysis/45_no_maccs.tsv'
no_gene_pred_file	= 'prediction/ablation-analysis/45_no_genes.tsv' 
no_assay_pred_file	= 'prediction/ablation-analysis/45_no_assays.tsv'
plot_title		= 'tox21-rt-viability-hepg2-p2'

## 1. Obtain performance metrics of all models  
roc_list = []  
# compute TPR, FPR, AUROC, 95% CI for the full model 
fu_pred_df = pd.read_csv(full_pred_file, sep = '\t', header = 0) 
fu_roc_fpr, fu_roc_tpr, _ = roc_curve(fu_pred_df.true_label.values, fu_pred_df.proba.values)
fu_auc = roc_auc_score(fu_pred_df.true_label.values, fu_pred_df.proba.values)
fu_auc_ci = compute_metric_ci_by_bootsrap(roc_auc_score, fu_pred_df.true_label.values, fu_pred_df.proba.values)
fu_label = 'GNN - full\n(AUC=' + str(round(fu_auc, 2)) + '±' + str(round(fu_auc_ci, 2)) + ')'
roc_list.append((fu_roc_fpr, fu_roc_tpr, fu_label))
# compute TPR, FPR, AUROC, 95% CI for the 'no structure' model 
ns_pred_df = pd.read_csv(no_struture_pred_file, sep = '\t', header = 0)   
ns_roc_fpr, ns_roc_tpr, _ = roc_curve(ns_pred_df.true_label.values, ns_pred_df.proba.values)
ns_auc = roc_auc_score(ns_pred_df.true_label.values, ns_pred_df.proba.values)
ns_auc_ci = compute_metric_ci_by_bootsrap(roc_auc_score, ns_pred_df.true_label.values, ns_pred_df.proba.values)
ns_label = 'GNN - no structure\n(AUC=' + str(round(ns_auc, 2)) + '±' + str(round(ns_auc_ci, 2)) + ')'
roc_list.append((ns_roc_fpr, ns_roc_tpr, ns_label))
# compute TPR, FPR, AUROC, 95% CI for the 'no gene' model 
ng_pred_df = pd.read_csv(no_gene_pred_file, sep = '\t', header = 0)
ng_roc_fpr, ng_roc_tpr, _ = roc_curve(ng_pred_df.true_label.values, ng_pred_df.proba.values)
ng_auc = roc_auc_score(ng_pred_df.true_label.values, ng_pred_df.proba.values)
ng_auc_ci = compute_metric_ci_by_bootsrap(roc_auc_score, ng_pred_df.true_label.values, ng_pred_df.proba.values)
ng_label = 'GNN - no gene\n(AUC=' + str(round(ng_auc, 2)) + '±' + str(round(ng_auc_ci, 2)) + ')'
roc_list.append((ng_roc_fpr, ng_roc_tpr, ng_label)) 
# compute TPR, FPR, AUROC, 95% CI for the 'no assay' model 
na_pred_df = pd.read_csv(no_assay_pred_file, sep = '\t', header = 0)
na_roc_fpr, na_roc_tpr, _ = roc_curve(na_pred_df.true_label.values, na_pred_df.proba.values)
na_auc = roc_auc_score(na_pred_df.true_label.values, na_pred_df.proba.values)
na_auc_ci = compute_metric_ci_by_bootsrap(roc_auc_score, na_pred_df.true_label.values, na_pred_df.proba.values)
na_label = 'GNN - no assay\n(AUC=' + str(round(na_auc, 2)) + '±' + str(round(na_auc_ci, 2)) + ')'
roc_list.append((na_roc_fpr, na_roc_tpr, na_label))

## 2. Make ROC curves 
# specify figure and font size
plt.figure(figsize = (6, 6))
plt.rc('font', size = 20)
plt.rc('axes', titlesize = 20)
plt.rc('axes', labelsize = 20)
plt.rc('xtick', labelsize = 15)
plt.rc('ytick', labelsize = 15)
plt.rc('legend', fontsize = 15)
# plot ROC curves of multiple classifiers 
plot_color = [plt.cm.Set1(0), plt.cm.Set1(3), plt.cm.Set1(4), plt.cm.Set1(6)]
for nrl in range(0, len(roc_list)):
	rl = roc_list[nrl]
	rl_fpr, rl_tpr, rl_label = rl
	plt.plot(rl_fpr, rl_tpr, lw = 2, label = rl_label, color = plot_color[nrl])
# plot baseline line of a random classifier (diagonal line)
plt.plot([0, 1], [0, 1], color = 'grey', lw = 1, linestyle='--')
# save plot
plt.xlim([-0.01, 1])
plt.ylim([0, 1.01])
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title(plot_title)
plt.legend(loc = 'lower right', frameon = False)
plt.tight_layout()
plt.savefig(plot_folder + plot_title + '_roc.pdf')
plt.close()
