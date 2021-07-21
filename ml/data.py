# !/usr/bin/env python
# created by Yun Hao @MooreLab 2019
# This script contains functions for data pre-processing. 


##  Module
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


## This function generates summary statistics of a dataset (number of samples)
def generate_data_summary(data_df, group, label):
	## 0. Input argument:
		# data_df: input dataset
		# group: name of group  
		# label: name of label column
	
	## 1. Obtain number of samples
	N_sample = data_df.shape[0]
	
	## 2. Output summary statistics
	uni_labels = sorted(data_df[label].unique())
	if uni_labels == [0, 1]:
		# number of positive/negative samples 
		N_positive = data_df[label].sum() 
		N_negative = N_sample - N_positive
		# output 
		out_dict = {'Group': group, 'N_samples': N_sample, 'N_positive_samples': N_positive, 'N_negative_samples': N_negative}	
	else:
		out_dict = {'Group': group, 'N_samples': N_sample}
		
	return out_dict

	
## This function can split a dataset into train set and test set  
def split_dataset_into_train_test(data_df, label, test_prop, task, output_pf):
	## 0. Input argument: 
		# data_df: input dataset  
		# label: name of label column
		# test_prop: proportion of test data  
		# task: type of supervised learning task: 'regression' or 'classification' 
		# output_pf: prefix of output file 
	
	## 1. Split whole dataset into train and test set 
	# separate feature and response  
	X, y = data_df.drop(label, axis = 1), data_df[label]
	# train/test split
	if task == 'classification':
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_prop, random_state = 0, stratify = y)
	if task == 'regression':
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_prop, random_state = 0)

	## 2. Output train and test datasets 
	# train set  
	train_df = pd.concat([X_train, y_train], axis = 1)
	train_df.to_csv(output_pf + '_train.tsv', sep = '\t')
	# test set 
	test_df = pd.concat([X_test, y_test], axis = 1)	
	test_df.to_csv(output_pf + '_test.tsv', sep = '\t')

	return 1


## This function can generate feature-response datasets for learning task from group-sample relationships, then split each dataset into train set and test set 
def generate_learning_dataset(relation_file, feature_file, min_N_sample, test_proportion, output_folder):
	## 0. Input argments:
		# relation_file: input file that contains group-sample relationships (three columns, 1: group, 2: sample, 3: label) 
		# feature_file: input file that contains computed features of samples
		# min_N_sample: minimum number of samples required 
		# test_proportion: proportion of test data   
		# output_folder: folder to store the output files 
	
	## 1. Obtain group-sample relationships 
	# read in input file 
	relation_df = pd.read_csv(relation_file, sep = '\t', header = 0)
	# obtain group names
	group_col = relation_df.columns[0]
	groups = relation_df[group_col].unique()
	# obtain name of the sample column 
	sample_col = relation_df.columns[1]
	# obtain name of the label(response) column
	label_col = relation_df.columns[2]
	# obtain type of task
	label_len = len(relation_df[label_col].unique())
	if label_len <= 2:
		task_type = 'classification'
		min_class_sample = int(min_N_sample/2)
	else:	
		task_type = 'regression'

	## 2. Obtain feature data
	# read in input file 
	feature_df = pd.read_csv(feature_file, sep = '\t', header = 0, index_col = 0)
	# obtain sample names
	feature_samples = feature_df.index

	## 3. Generate sample-feature dataset  
	# iterate by group 
	groups_summary = []
	for group in groups:
		# obtain samples that belong to the group 
		group_relation_df = relation_df[relation_df[group_col] == group]
		sample_names = []
		# iterate by sample name 
		for index, row in group_relation_df.iterrows():
			sample_name = row[sample_col]
			# process str type 
			if isinstance(sample_name, str) == True:
				# add '_' between words in a sample name  
				sample_name = '_'.join(sample_name.split())
			sample_names.append(sample_name)
		# merge two data frames to obtain whole dateset 
		group_relation_df.index = sample_names
		combine_df = pd.merge(feature_df, group_relation_df[label_col], left_index = True, right_index = True)
		# check the number of samples, make sure it satisfied the minimum requirement  
		sample_check = 0
		if combine_df.shape[0] >= min_N_sample:
			# for classification tasks, each class needs to satisfy a minimum requirement 
			if task_type == 'classification':
				if (combine_df[label_col].value_counts().values >= min_class_sample).sum() == label_len:
					sample_check = 1
			# for regression tasks, the whole dataset needs to satisfy overall minimum requirement
			if task_type == 'regression':
				sample_check = 1
		# output datasets   
		if sample_check == 1:
			output_file = output_folder + '_' + group + '_whole_data.tsv'
			combine_df.to_csv(output_file, sep = '\t', index = True, float_format = '%.5f')	
			# get summary statistics 
			group_summary = generate_data_summary(combine_df, group, label_col)
			groups_summary.append(group_summary)
			# split whole data into train set and test set  
			combine_df_split = split_dataset_into_train_test(combine_df, label_col, test_proportion, task_type, output_file)
		
	## 4. Output summary statisticss
	groups_summary_df = pd.DataFrame(groups_summary)
	output_sum_file = output_folder + '_whole_data_summary.tsv'
	groups_summary_df.to_csv(output_sum_file, sep = '\t', index = False)
	
	return 1 
