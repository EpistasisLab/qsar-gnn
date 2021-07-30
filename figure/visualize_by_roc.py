# !/usr/bin/env python
# created by Yun Hao @MooreLab 2019
# This script contains codes that visualize ROC curves comparison across three methods


## Module
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt


## 0. Input and output files 
rf_perf_file    	= 'performance/compound_structure_fingerprint_maccs_tox21_simple_randomforest_optimal_performance_summary_by_training_log_loss.tsv'
gb_perf_file	    	= 'performance/compound_structure_fingerprint_maccs_tox21_simple_xgboost_optimal_performance_summary_by_training_log_loss.tsv'
gnn_perf_file		= 'performance/tox21_gnn_performance_summary.tsv'
gnn_map_file		= 'performance/nc_results_with_names.tsv'
prediction_folder	= 'prediction/'
plot_roc_folder		= 'plot/roc/performance_comparison_'

## 1. Obtain performance metrics of three methods 
rf_perf_df = pd.read_csv(rf_perf_file, sep = '\t', header = 0)
gb_perf_df = pd.read_csv(gb_perf_file, sep = '\t', header = 0)
gnn_perf_df = pd.read_csv(gnn_perf_file, sep = '\t', header = 0)
gnn_map_df = pd.read_csv(gnn_map_file, sep = '\t', header = 0)

## 2. Make ROC curves to compare three methods 
plot_color = [plt.cm.Set1(0), plt.cm.Set1(1), plt.cm.Set1(2)]
for gdv in gnn_perf_df.dataset_name.values:
	print(gdv)
	roc_list = []  
	# compute TPR, FPR for graph neural network predictions
	gnn_query_row = gnn_perf_df[gnn_perf_df.dataset_name == gdv]
	gnn_map_row = gnn_map_df[gnn_map_df.assay_name == gdv]
	gnn_pred_file = prediction_folder + str(gnn_map_row.assay_index.values[0]) + '.tsv'
	gnn_pred_df = pd.read_csv(gnn_pred_file, sep = '\t', header = 0)
	gnn_roc_fpr, gnn_roc_tpr, _ = roc_curve(gnn_pred_df.true_label.values, gnn_pred_df.proba.values)
	gnn_label = 'GNN - full\n(AUC=' + str(round(gnn_query_row.testing_auc.values[0], 2)) + '±' + str(round(gnn_query_row.testing_auc_ci.values[0], 2)) + ')'
	roc_list.append((gnn_roc_fpr, gnn_roc_tpr, gnn_label))
	# compute TPR, FPR for random forest predictions  
	rf_query_row = rf_perf_df[rf_perf_df.dataset_name == gdv]
	rf_pred_file = prediction_folder + 'compound_structure_fingerprint_maccs_' + gdv + '_rf_pred.tsv'
	rf_pred_df = pd.read_csv(rf_pred_file, sep = '\t', header = 0) 
	rf_roc_fpr, rf_roc_tpr, _ = roc_curve(rf_pred_df.true_label.values, rf_pred_df.proba.values)
	rf_label = 'Random Forest\n(AUC=' + str(round(rf_query_row.testing_auc.values[0], 2)) + '±' + str(round(rf_query_row.testing_auc_ci.values[0], 2)) + ')'
	roc_list.append((rf_roc_fpr, rf_roc_tpr, rf_label))
	# compute TPR, FPR for gradient boosting predictions  
	gb_query_row = gb_perf_df[gb_perf_df.dataset_name == gdv]
	gb_pred_file = prediction_folder + 'compound_structure_fingerprint_maccs_' + gdv + '_gb_pred.tsv'
	gb_pred_df = pd.read_csv(gb_pred_file, sep = '\t', header = 0)
	gb_roc_fpr, gb_roc_tpr, _ = roc_curve(gb_pred_df.true_label.values, gb_pred_df.proba.values)
	gb_label = 'Gradient Boosting\n(AUC=' + str(round(gb_query_row.testing_auc.values[0], 2)) + '±' + str(round(gb_query_row.testing_auc_ci.values[0], 2)) + ')'
	roc_list.append((gb_roc_fpr, gb_roc_tpr, gb_label))
	# specify figure and font size
	plt.figure(figsize = (6, 6))
	plt.rc('font', size = 20)
	plt.rc('axes', titlesize = 20)
	plt.rc('axes', labelsize = 20)
	plt.rc('xtick', labelsize = 15)
	plt.rc('ytick', labelsize = 15)
	plt.rc('legend', fontsize = 15)
	# plot ROC curves of multiple classifiers 
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
	plt.title(gdv)
	plt.legend(loc = 'lower right', frameon = False)
	plt.tight_layout()
	plt.savefig(plot_roc_folder + gdv + '_roc.pdf')
	plt.close()
