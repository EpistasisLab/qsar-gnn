# !/usr/bin/env python
# created by Yun Hao @MooreLab 2021
# This script contains codes that visualize performance metrics comparison across three methods by violinplot


## Modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


## 0. Input and output files 
rf_perf_file		= 'performance/compound_structure_fingerprint_maccs_tox21_simple_randomforest_optimal_performance_summary_by_training_log_loss.tsv'
gb_perf_file		= 'performance/compound_structure_fingerprint_maccs_tox21_simple_xgboost_optimal_performance_summary_by_training_log_loss.tsv'
gnn_perf_file		= 'performance/tox21_gnn_performance_summary.tsv'
out_perf_file		= 'performance/tox21_all_performance_summary.tsv'
plot_violin_file	= 'plot/performance_comparison_violinplot'

## 1. Obtain performance metrics of all methods
# Random forest 
rf_perf_df = pd.read_csv(rf_perf_file, sep = '\t', header = 0)
interest_cols = ['dataset_name', 'testing_auc', 'testing_auc_ci', 'testing_f1', 'testing_f1_ci']
rf_perf_df1 = rf_perf_df[interest_cols]
rf_perf_df1.columns = ['dataset_name', 'rf_testing_auc', 'rf_testing_auc_ci', 'rf_testing_f1', 'rf_testing_f1_ci']
# Gradient boosting 
gb_perf_df = pd.read_csv(gb_perf_file, sep = '\t', header = 0)
gb_perf_df1 = gb_perf_df[interest_cols]
gb_perf_df1.columns = ['dataset_name', 'gb_testing_auc', 'gb_testing_auc_ci', 'gb_testing_f1', 'gb_testing_f1_ci']
# Graph neural network
gnn_perf_df = pd.read_csv(gnn_perf_file, sep = '\t', header = 0)
gnn_perf_df.columns = ['dataset_name', 'gnn_testing_auc', 'gnn_testing_auc_ci', 'gnn_testing_f1', 'gnn_testing_f1_ci']
# Merge all three metric data frames 
all_perf_df1 = pd.merge(rf_perf_df1, gb_perf_df1, on = 'dataset_name')
all_perf_df = pd.merge(all_perf_df1, gnn_perf_df, on = 'dataset_name')
all_perf_df.to_csv(out_perf_file, sep = '\t', float_format = '%.5f', index = False)

## 2. Make violin plot to compare F1 scores across three methods  
# obtain data frame containing F1 scores of three methods 
N_assay = all_perf_df.shape[0]
rf_f1_df = pd.DataFrame({'Classification method': np.repeat('Random\nforest', N_assay), 'F1 score': all_perf_df.rf_testing_f1.values})
gb_f1_df = pd.DataFrame({'Classification method': np.repeat('Gradient\nboosting', N_assay), 'F1 score': all_perf_df.gb_testing_f1.values})
gnn_f1_df = pd.DataFrame({'Classification method': np.repeat('Graph neural\nnetwork', N_assay), 'F1 score': all_perf_df.gnn_testing_f1.values})
f1_df = pd.concat([rf_f1_df, gb_f1_df, gnn_f1_df])
# specify figure and font size of boxplot 
plt.figure(figsize = (8, 6))
plt.rc('font', size = 25)
plt.rc('axes', titlesize = 30)
plt.rc('axes', labelsize = 30)
plt.rc('xtick', labelsize = 20)
plt.rc('ytick', labelsize = 30)
plt.rc('legend', fontsize = 25)
# make violinplot
ax = sns.violinplot(x = "Classification method", y = "F1 score", data = f1_df)
plt.tight_layout()
plt.savefig(plot_violin_file + '_f1.pdf')
plt.close()

## 3. Make violin plot to compare AUROCs across three methods   
# obtain data frame containing AUROCs of three methods 
rf_auc_df = pd.DataFrame({'Classification method': np.repeat('Random\nforest', N_assay), 'AUROC': all_perf_df.rf_testing_auc.values})
gb_auc_df = pd.DataFrame({'Classification method': np.repeat('Gradient\nboosting', N_assay), 'AUROC': all_perf_df.gb_testing_auc.values})
gnn_auc_df = pd.DataFrame({'Classification method': np.repeat('Graph neural\nnetwork', N_assay), 'AUROC': all_perf_df.gnn_testing_auc.values})
auc_df = pd.concat([rf_auc_df, gb_auc_df, gnn_auc_df])
# specify figure and font size of boxplot 
plt.figure(figsize = (8, 6))
plt.rc('font', size = 25)
plt.rc('axes', titlesize = 30)
plt.rc('axes', labelsize = 30)
plt.rc('xtick', labelsize = 20)
plt.rc('ytick', labelsize = 30)
plt.rc('legend', fontsize = 25)
# make violinplot
ax = sns.violinplot(x = 'Classification method', y = 'AUROC', data = auc_df)
plt.tight_layout()
plt.savefig(plot_violin_file + '_auc.pdf')
plt.close()

